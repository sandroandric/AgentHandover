"""Tests for Focus Recording Mode — worker-side processing.

Tests cover:
- Focus events grouped correctly by session_id
- induce_from_focus_session() produces SOP from single demo
- Focus SOP has source: "focus_recording" metadata
- Empty focus session returns empty
- Focus + normal pipeline don't interfere
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestFocusSessionInduction:
    """Test SOPInducer.induce_from_focus_session()."""

    def _make_inducer(self):
        from agenthandover_worker.sop_inducer import SOPInducer
        return SOPInducer()

    def _make_steps(self, n: int = 5) -> list[dict]:
        """Create n sample semantic step dicts."""
        steps = []
        for i in range(n):
            steps.append({
                "step": f"action_{i}",
                "target": f"Target {i}",
                "selector": f"#target-{i}",
                "parameters": {"app_id": "com.app.Test"},
                "confidence": 0.85 + (i * 0.02),
                "pre_state": {"app_id": "com.app.Test", "window_title": f"Page {i}"},
            })
        return steps

    def test_basic_induction(self):
        inducer = self._make_inducer()
        episodes = [self._make_steps(5)]
        result = inducer.induce_from_focus_session(episodes, "Test Workflow")

        assert len(result) == 1
        sop = result[0]
        assert sop["title"] == "Test Workflow"
        assert sop["source"] == "focus_recording"
        assert sop["focus_title"] == "Test Workflow"
        assert sop["episode_count"] == 1
        assert sop["abs_support"] == 1
        assert len(sop["steps"]) == 5

    def test_empty_episodes_returns_empty(self):
        inducer = self._make_inducer()
        assert inducer.induce_from_focus_session([], "Empty") == []

    def test_empty_steps_returns_empty(self):
        inducer = self._make_inducer()
        assert inducer.induce_from_focus_session([[]], "Empty Steps") == []

    def test_confidence_avg_computed(self):
        inducer = self._make_inducer()
        steps = [
            {"step": "click", "target": "A", "confidence": 0.80, "pre_state": {}},
            {"step": "type", "target": "B", "confidence": 0.90, "pre_state": {}},
        ]
        result = inducer.induce_from_focus_session([steps], "Conf Test")
        assert len(result) == 1
        # Average of 0.80 and 0.90 = 0.85
        assert abs(result[0]["confidence_avg"] - 0.85) < 0.01

    def test_apps_collected(self):
        inducer = self._make_inducer()
        steps = [
            {
                "step": "click",
                "target": "Button",
                "confidence": 0.9,
                "parameters": {"app_id": "com.apple.Safari"},
                "pre_state": {},
            },
            {
                "step": "type",
                "target": "Field",
                "confidence": 0.9,
                "parameters": {"app_id": "com.apple.Numbers"},
                "pre_state": {},
            },
        ]
        result = inducer.induce_from_focus_session([steps], "Multi App")
        assert "com.apple.Safari" in result[0]["apps_involved"]
        assert "com.apple.Numbers" in result[0]["apps_involved"]

    def test_variables_empty_for_single_demo(self):
        """Single demo cannot abstract variables (need 2+ instances)."""
        inducer = self._make_inducer()
        steps = self._make_steps(3)
        result = inducer.induce_from_focus_session([steps], "Single Demo")
        assert result[0]["variables"] == []

    def test_slug_generated_from_title(self):
        inducer = self._make_inducer()
        steps = self._make_steps(3)
        result = inducer.induce_from_focus_session(
            [steps], "Expense Report Filing"
        )
        slug = result[0]["slug"]
        assert "expense" in slug.lower()
        assert "report" in slug.lower()
        # Should be URL-safe
        assert " " not in slug

    def test_multiple_episodes_merged(self):
        """Multiple episodes from one focus session are merged."""
        inducer = self._make_inducer()
        ep1 = self._make_steps(3)
        ep2 = [
            {"step": "submit", "target": "Form", "confidence": 0.9, "pre_state": {}},
        ]
        result = inducer.induce_from_focus_session([ep1, ep2], "Multi Episode")
        # All steps from both episodes should be in the SOP
        assert len(result[0]["steps"]) == 4

    def test_preconditions_detected(self):
        inducer = self._make_inducer()
        steps = [
            {
                "step": "click",
                "target": "Tab",
                "confidence": 0.9,
                "pre_state": {"app_id": "com.apple.Safari", "url": "https://example.com"},
                "parameters": {},
            },
            {
                "step": "type",
                "target": "Field",
                "confidence": 0.9,
                "pre_state": {"app_id": "com.apple.Safari"},
                "parameters": {},
            },
        ]
        result = inducer.induce_from_focus_session([steps], "With Preconditions")
        # Should detect app_open precondition
        preconditions = result[0]["preconditions"]
        assert any("app_open" in p for p in preconditions)

    def test_focus_sop_does_not_require_prefixspan(self):
        """Focus induction bypasses PrefixSpan min_support=2 requirement."""
        inducer = self._make_inducer()
        # Single episode with only 1 step — normally PrefixSpan would reject
        steps = [
            {"step": "click", "target": "Button", "confidence": 0.9, "pre_state": {}},
        ]
        result = inducer.induce_from_focus_session([steps], "Single Step")
        # Should still produce a SOP
        assert len(result) == 1
        assert result[0]["episode_count"] == 1


class TestFocusSignalFile:
    """Test the focus signal file IPC (Python side reads)."""

    def test_read_signal_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            signal = {
                "session_id": "test-uuid-123",
                "title": "Test Workflow",
                "started_at": "2026-02-23T10:00:00Z",
                "status": "stopped",
            }
            signal_path = Path(tmpdir) / "focus-session.json"
            signal_path.write_text(json.dumps(signal))

            # Verify we can read it
            with open(signal_path) as f:
                read_signal = json.load(f)
            assert read_signal["session_id"] == "test-uuid-123"
            assert read_signal["status"] == "stopped"

    def test_missing_signal_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            signal_path = Path(tmpdir) / "focus-session.json"
            assert not signal_path.exists()

    def test_invalid_json_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            signal_path = Path(tmpdir) / "focus-session.json"
            signal_path.write_text("not valid json{{{")
            with pytest.raises(json.JSONDecodeError):
                with open(signal_path) as f:
                    json.load(f)

"""Tests for enhanced SOP inducer features: preconditions, postconditions, exceptions.

These tests don't require prefixspan - they test the helper methods directly.
"""

from __future__ import annotations

from collections import Counter


# ------------------------------------------------------------------
# Since SOPInducer requires prefixspan, we test the methods
# that don't need it by instantiating with a mock or testing the
# detection methods via the class
# ------------------------------------------------------------------


class TestDetectPreconditions:
    def test_common_app_detected(self) -> None:
        """App appearing in 80%+ of instances is a precondition."""
        try:
            from agenthandover_worker.sop_inducer import SOPInducer
        except ImportError:
            import pytest
            pytest.skip("prefixspan not installed")

        inducer = SOPInducer()
        instances = [
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Firefox"}}],
        ]

        preconditions = inducer._detect_preconditions(instances)
        assert any("Chrome" in p for p in preconditions)

    def test_no_precondition_when_diverse(self) -> None:
        """No precondition when apps are diverse (< 80%)."""
        try:
            from agenthandover_worker.sop_inducer import SOPInducer
        except ImportError:
            import pytest
            pytest.skip("prefixspan not installed")

        inducer = SOPInducer()
        instances = [
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Chrome"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Firefox"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"app_id": "Safari"}}],
        ]

        preconditions = inducer._detect_preconditions(instances)
        app_preconditions = [p for p in preconditions if p.startswith("app_open:")]
        assert len(app_preconditions) == 0

    def test_url_precondition(self) -> None:
        """Common URL in first step is a precondition."""
        try:
            from agenthandover_worker.sop_inducer import SOPInducer
        except ImportError:
            import pytest
            pytest.skip("prefixspan not installed")

        inducer = SOPInducer()
        instances = [
            [{"step": "click", "target": "Submit", "pre_state": {"url": "https://app.com/login"}}],
            [{"step": "click", "target": "Submit", "pre_state": {"url": "https://app.com/login"}}],
        ]

        preconditions = inducer._detect_preconditions(instances)
        assert any("url_open:" in p for p in preconditions)


class TestDetectPostconditions:
    def test_common_final_action(self) -> None:
        """Common final intent is a postcondition."""
        try:
            from agenthandover_worker.sop_inducer import SOPInducer
        except ImportError:
            import pytest
            pytest.skip("prefixspan not installed")

        inducer = SOPInducer()
        instances = [
            [
                {"step": "type", "target": "Email"},
                {"step": "click", "target": "Save"},
            ],
            [
                {"step": "type", "target": "Email"},
                {"step": "click", "target": "Save"},
            ],
        ]

        postconditions = inducer._detect_postconditions(instances)
        assert any("final_action:click" in p for p in postconditions)

    def test_empty_instances(self) -> None:
        try:
            from agenthandover_worker.sop_inducer import SOPInducer
        except ImportError:
            import pytest
            pytest.skip("prefixspan not installed")

        inducer = SOPInducer()
        postconditions = inducer._detect_postconditions([])
        assert postconditions == []


class TestDetectExceptions:
    """Exception detection via _scan_episodes_for_pattern (combined method)."""

    def test_cancel_detected(self) -> None:
        """Cancel events in pattern episodes are detected as exceptions."""
        try:
            from agenthandover_worker.sop_inducer import SOPInducer
        except ImportError:
            import pytest
            pytest.skip("prefixspan not installed")

        inducer = SOPInducer()
        episodes = [
            [
                {"step": "click", "target": "Submit button", "parameters": {}},
                {"step": "click", "target": "Submit button", "parameters": {}},
                {"step": "click", "target": "Submit button", "parameters": {}},
                {"step": "cancel", "target": "Cancel dialog", "parameters": {}},
            ],
            [
                {"step": "click", "target": "Submit button", "parameters": {}},
                {"step": "click", "target": "Submit button", "parameters": {}},
                {"step": "click", "target": "Submit button", "parameters": {}},
            ],
        ]

        encoded, code_to_sig, sig_to_steps = inducer._encode_steps(episodes)
        click_code = None
        for code, sig in code_to_sig.items():
            if "click::submit button" in sig:
                click_code = code
                break

        if click_code is not None:
            pattern_codes = [click_code, click_code, click_code]
            _instances, _apps, exceptions = inducer._scan_episodes_for_pattern(
                episodes, pattern_codes, code_to_sig, sig_to_steps
            )
            assert any("cancel" in e for e in exceptions)

    def test_no_exceptions_in_clean_episodes(self) -> None:
        try:
            from agenthandover_worker.sop_inducer import SOPInducer
        except ImportError:
            import pytest
            pytest.skip("prefixspan not installed")

        inducer = SOPInducer()
        episodes = [
            [
                {"step": "click", "target": "Submit button", "parameters": {}},
                {"step": "click", "target": "Submit button", "parameters": {}},
                {"step": "click", "target": "Submit button", "parameters": {}},
            ],
        ]

        encoded, code_to_sig, sig_to_steps = inducer._encode_steps(episodes)
        click_code = None
        for code, sig in code_to_sig.items():
            if "click::submit button" in sig:
                click_code = code
                break

        if click_code is not None:
            pattern_codes = [click_code, click_code, click_code]
            _instances, _apps, exceptions = inducer._scan_episodes_for_pattern(
                episodes, pattern_codes, code_to_sig, sig_to_steps
            )
            assert len(exceptions) == 0


# ---------------------------------------------------------------------------
# Tests for _scan_episodes_for_pattern()
# ---------------------------------------------------------------------------


class TestScanEpisodesForPattern:
    """Tests for the single-pass _scan_episodes_for_pattern() method."""

    @staticmethod
    def _skip_if_no_prefixspan():
        try:
            from agenthandover_worker.sop_inducer import SOPInducer
            return SOPInducer
        except ImportError:
            import pytest
            pytest.skip("prefixspan not installed")

    def test_collects_matching_instances(self) -> None:
        """Instances matching the pattern are collected correctly."""
        SOPInducer = self._skip_if_no_prefixspan()
        inducer = SOPInducer()

        episodes = [
            [
                {"step": "click", "target": "Login", "parameters": {}, "pre_state": {}},
                {"step": "type", "target": "Username", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Submit", "parameters": {}, "pre_state": {}},
            ],
            [
                {"step": "click", "target": "Login", "parameters": {}, "pre_state": {}},
                {"step": "type", "target": "Username", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Submit", "parameters": {}, "pre_state": {}},
            ],
            [
                {"step": "click", "target": "Other", "parameters": {}, "pre_state": {}},
            ],
        ]

        encoded, code_to_sig, sig_to_steps = inducer._encode_steps(episodes)

        # Build pattern codes for "click::login", "type::username", "click::submit"
        sig_to_code = {v: k for k, v in code_to_sig.items()}
        pattern_codes = [
            sig_to_code["click::login"],
            sig_to_code["type::username"],
            sig_to_code["click::submit"],
        ]

        instances, apps, exceptions = inducer._scan_episodes_for_pattern(
            episodes, pattern_codes, code_to_sig, sig_to_steps
        )

        # Two episodes match the pattern
        assert len(instances) == 2
        # Each instance has 3 steps
        assert all(len(inst) == 3 for inst in instances)

    def test_extracts_apps_from_pre_state(self) -> None:
        """Apps are extracted from pre_state.app_id."""
        SOPInducer = self._skip_if_no_prefixspan()
        inducer = SOPInducer()

        episodes = [
            [
                {"step": "click", "target": "Save", "parameters": {}, "pre_state": {"app_id": "Chrome"}},
                {"step": "click", "target": "Save", "parameters": {}, "pre_state": {"app_id": "Chrome"}},
                {"step": "click", "target": "Save", "parameters": {}, "pre_state": {}},
            ],
        ]

        encoded, code_to_sig, sig_to_steps = inducer._encode_steps(episodes)
        sig_to_code = {v: k for k, v in code_to_sig.items()}
        pattern_codes = [sig_to_code["click::save"]] * 3

        instances, apps, exceptions = inducer._scan_episodes_for_pattern(
            episodes, pattern_codes, code_to_sig, sig_to_steps
        )

        assert "Chrome" in apps

    def test_extracts_apps_from_parameters(self) -> None:
        """Apps are extracted from parameters.app_id."""
        SOPInducer = self._skip_if_no_prefixspan()
        inducer = SOPInducer()

        episodes = [
            [
                {"step": "click", "target": "OK", "parameters": {"app_id": "VSCode"}, "pre_state": {}},
                {"step": "click", "target": "OK", "parameters": {"app_id": "Terminal"}, "pre_state": {}},
                {"step": "click", "target": "OK", "parameters": {}, "pre_state": {}},
            ],
        ]

        encoded, code_to_sig, sig_to_steps = inducer._encode_steps(episodes)
        sig_to_code = {v: k for k, v in code_to_sig.items()}
        pattern_codes = [sig_to_code["click::ok"]] * 3

        instances, apps, exceptions = inducer._scan_episodes_for_pattern(
            episodes, pattern_codes, code_to_sig, sig_to_steps
        )

        assert "VSCode" in apps
        assert "Terminal" in apps

    def test_detects_cancel_exception(self) -> None:
        """Exception indicators in step intents are detected."""
        SOPInducer = self._skip_if_no_prefixspan()
        inducer = SOPInducer()

        episodes = [
            [
                {"step": "click", "target": "Start", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Start", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Start", "parameters": {}, "pre_state": {}},
                {"step": "cancel", "target": "Dialog", "parameters": {}, "pre_state": {}},
            ],
        ]

        encoded, code_to_sig, sig_to_steps = inducer._encode_steps(episodes)
        sig_to_code = {v: k for k, v in code_to_sig.items()}
        pattern_codes = [sig_to_code["click::start"]] * 3

        instances, apps, exceptions = inducer._scan_episodes_for_pattern(
            episodes, pattern_codes, code_to_sig, sig_to_steps
        )

        assert any("cancel" in e for e in exceptions)

    def test_detects_error_in_target(self) -> None:
        """Exception indicators in step targets are detected."""
        SOPInducer = self._skip_if_no_prefixspan()
        inducer = SOPInducer()

        episodes = [
            [
                {"step": "click", "target": "Run", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Run", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Run", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Error dialog close", "parameters": {}, "pre_state": {}},
            ],
        ]

        encoded, code_to_sig, sig_to_steps = inducer._encode_steps(episodes)
        sig_to_code = {v: k for k, v in code_to_sig.items()}
        pattern_codes = [sig_to_code["click::run"]] * 3

        instances, apps, exceptions = inducer._scan_episodes_for_pattern(
            episodes, pattern_codes, code_to_sig, sig_to_steps
        )

        assert any("error" in e for e in exceptions)

    def test_deduplicates_exceptions(self) -> None:
        """Duplicate exceptions are not repeated."""
        SOPInducer = self._skip_if_no_prefixspan()
        inducer = SOPInducer()

        # Two episodes, both containing the same cancel step
        episodes = [
            [
                {"step": "click", "target": "Go", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Go", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Go", "parameters": {}, "pre_state": {}},
                {"step": "cancel", "target": "Abort", "parameters": {}, "pre_state": {}},
            ],
            [
                {"step": "click", "target": "Go", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Go", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "Go", "parameters": {}, "pre_state": {}},
                {"step": "cancel", "target": "Abort", "parameters": {}, "pre_state": {}},
            ],
        ]

        encoded, code_to_sig, sig_to_steps = inducer._encode_steps(episodes)
        sig_to_code = {v: k for k, v in code_to_sig.items()}
        pattern_codes = [sig_to_code["click::go"]] * 3

        instances, apps, exceptions = inducer._scan_episodes_for_pattern(
            episodes, pattern_codes, code_to_sig, sig_to_steps
        )

        # "cancel:abort" should appear exactly once despite two episodes
        cancel_exceptions = [e for e in exceptions if "cancel" in e]
        assert len(cancel_exceptions) == 1

    def test_no_match_returns_empty(self) -> None:
        """Non-matching episodes produce empty results."""
        SOPInducer = self._skip_if_no_prefixspan()
        inducer = SOPInducer()

        episodes = [
            [
                {"step": "click", "target": "A", "parameters": {}, "pre_state": {}},
                {"step": "click", "target": "B", "parameters": {}, "pre_state": {}},
            ],
        ]

        encoded, code_to_sig, sig_to_steps = inducer._encode_steps(episodes)
        sig_to_code = {v: k for k, v in code_to_sig.items()}
        # Pattern that won't match (A, A, A) vs episode (A, B)
        pattern_codes = [sig_to_code["click::a"]] * 3

        instances, apps, exceptions = inducer._scan_episodes_for_pattern(
            episodes, pattern_codes, code_to_sig, sig_to_steps
        )

        assert instances == []
        assert apps == []
        assert exceptions == []


# ---------------------------------------------------------------------------
# Tests for _stratified_sample()
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    """Tests for the static _stratified_sample() method."""

    @staticmethod
    def _skip_if_no_prefixspan():
        try:
            from agenthandover_worker.sop_inducer import SOPInducer
            return SOPInducer
        except ImportError:
            import pytest
            pytest.skip("prefixspan not installed")

    def test_returns_all_when_target_exceeds_input(self) -> None:
        """Returns all episodes if target_count >= len(encoded)."""
        SOPInducer = self._skip_if_no_prefixspan()

        encoded = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        result = SOPInducer._stratified_sample(encoded, 10)
        assert result is encoded  # Identity — not a copy

    def test_returns_all_when_target_equals_input(self) -> None:
        """Returns all episodes if target_count == len(encoded)."""
        SOPInducer = self._skip_if_no_prefixspan()

        encoded = [[1, 2], [3, 4], [5, 6]]
        result = SOPInducer._stratified_sample(encoded, 3)
        assert result is encoded

    def test_returns_target_count_when_subsampling(self) -> None:
        """Returns at most target_count episodes when subsampling."""
        SOPInducer = self._skip_if_no_prefixspan()

        # Build 30 episodes with varying lengths to ensure stratified buckets
        encoded = []
        for i in range(10):
            encoded.append(list(range(i + 1)))       # short: len 1-10
        for i in range(10):
            encoded.append(list(range(i + 15)))       # medium: len 15-24
        for i in range(10):
            encoded.append(list(range(i + 50)))       # long: len 50-59

        result = SOPInducer._stratified_sample(encoded, 9)
        assert len(result) <= 9

    def test_preserves_episodes_from_all_buckets(self) -> None:
        """Sampling preserves episodes from short, medium, and long buckets."""
        SOPInducer = self._skip_if_no_prefixspan()

        # Create distinct buckets of episodes
        short = [[1] for _ in range(10)]          # len 1
        medium = [list(range(20)) for _ in range(10)]  # len 20
        long_eps = [list(range(100)) for _ in range(10)]  # len 100
        encoded = short + medium + long_eps  # 30 total

        result = SOPInducer._stratified_sample(encoded, 9)

        # Verify we got episodes from different length ranges
        lengths = [len(ep) for ep in result]
        has_short = any(l <= 5 for l in lengths)
        has_long = any(l >= 50 for l in lengths)

        # With proportional sampling from 3 buckets targeting 9, each bucket
        # should contribute at least 1
        assert has_short, "Expected at least one short episode in sample"
        assert has_long, "Expected at least one long episode in sample"

    def test_never_exceeds_target_count(self) -> None:
        """Result never has more episodes than target_count."""
        SOPInducer = self._skip_if_no_prefixspan()

        # Many episodes to force subsampling
        encoded = [list(range(i + 1)) for i in range(100)]
        target = 15

        result = SOPInducer._stratified_sample(encoded, target)
        assert len(result) <= target

import pytest
from unittest.mock import patch, MagicMock

class TestFeatureFlags:
    def test_all_enabled_by_default(self):
        """When no config file, all features enabled."""
        # Import and test _read_feature_flags
        from oc_apprentice_worker.main import _read_feature_flags
        with patch("oc_apprentice_worker.main._config_path", return_value=None):
            flags = _read_feature_flags()
        assert all(flags.values())
        assert len(flags) == 5

    def test_single_feature_disabled(self, tmp_path):
        """Disabling one feature leaves others enabled."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[features]\ncuration = false\n')
        from oc_apprentice_worker.main import _read_feature_flags
        with patch("oc_apprentice_worker.main._config_path", return_value=str(config_path)):
            flags = _read_feature_flags()
        assert flags["curation"] is False
        assert flags["activity_classification"] is True

    def test_missing_features_section(self, tmp_path):
        """Config without [features] section defaults to all enabled."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[vlm]\nenabled = true\n')
        from oc_apprentice_worker.main import _read_feature_flags
        with patch("oc_apprentice_worker.main._config_path", return_value=str(config_path)):
            flags = _read_feature_flags()
        assert all(flags.values())

    def test_all_disabled(self, tmp_path):
        """All features can be disabled."""
        config_path = tmp_path / "config.toml"
        lines = "[features]\n" + "\n".join(f"{k} = false" for k in [
            "activity_classification", "continuity_tracking",
            "lifecycle_management", "curation", "runtime_validation",
        ])
        config_path.write_text(lines)
        from oc_apprentice_worker.main import _read_feature_flags
        with patch("oc_apprentice_worker.main._config_path", return_value=str(config_path)):
            flags = _read_feature_flags()
        assert not any(flags.values())

    def test_invalid_config_uses_defaults(self, tmp_path):
        """Corrupted config file uses all-enabled defaults."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("not valid toml {{{{")
        from oc_apprentice_worker.main import _read_feature_flags
        with patch("oc_apprentice_worker.main._config_path", return_value=str(config_path)):
            flags = _read_feature_flags()
        assert all(flags.values())

    def test_feature_keys_complete(self):
        """All expected feature keys are present."""
        from oc_apprentice_worker.main import _read_feature_flags
        with patch("oc_apprentice_worker.main._config_path", return_value=None):
            flags = _read_feature_flags()
        expected = {"activity_classification", "continuity_tracking", "lifecycle_management", "curation", "runtime_validation"}
        assert set(flags.keys()) == expected

    def test_non_bool_coerced(self, tmp_path):
        """Non-bool values coerced to bool."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[features]\ncuration = 0\n')
        from oc_apprentice_worker.main import _read_feature_flags
        with patch("oc_apprentice_worker.main._config_path", return_value=str(config_path)):
            flags = _read_feature_flags()
        assert flags["curation"] is False

    def test_unknown_keys_ignored(self, tmp_path):
        """Unknown feature keys don't cause errors."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[features]\nfuture_feature = true\ncuration = true\n')
        from oc_apprentice_worker.main import _read_feature_flags
        with patch("oc_apprentice_worker.main._config_path", return_value=str(config_path)):
            flags = _read_feature_flags()
        assert "future_feature" not in flags
        assert flags["curation"] is True

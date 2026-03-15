"""Configuration validator for config.toml.

Validates config values on startup so invalid settings are logged
as warnings rather than silently falling back to defaults.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConfigIssue:
    section: str
    key: str
    severity: str    # "error", "warning"
    message: str

class ConfigValidator:
    """Validate config.toml values against expected schemas."""

    def validate(self, config: dict) -> list[ConfigIssue]:
        """Validate all config sections. Returns list of issues (empty = valid)."""
        issues: list[ConfigIssue] = []
        if "vlm" in config:
            issues.extend(self.validate_vlm_section(config["vlm"]))
        if "knowledge" in config:
            issues.extend(self.validate_knowledge_section(config["knowledge"]))
        if "trust" in config:
            issues.extend(self.validate_trust_section(config["trust"]))
        if "privacy" in config:
            issues.extend(self.validate_privacy_section(config["privacy"]))
        if "features" in config:
            issues.extend(self.validate_features_section(config["features"]))
        return issues

    def validate_vlm_section(self, vlm: dict) -> list[ConfigIssue]:
        issues = []
        # Model names should be non-empty strings
        for key in ("annotation_model", "sop_model"):
            val = vlm.get(key)
            if val is not None and (not isinstance(val, str) or not val.strip()):
                issues.append(ConfigIssue("vlm", key, "warning", f"{key} should be a non-empty model name"))
        # Numeric bounds
        for key, lo, hi in [("max_jobs_per_day", 1, 10000), ("max_queue_size", 1, 50000), ("max_compute_minutes_per_day", 1, 1440)]:
            val = vlm.get(key)
            if val is not None and (not isinstance(val, (int, float)) or val < lo or val > hi):
                issues.append(ConfigIssue("vlm", key, "warning", f"{key} should be between {lo} and {hi}"))
        return issues

    def validate_knowledge_section(self, kb: dict) -> list[ConfigIssue]:
        issues = []
        port = kb.get("query_api_port")
        if port is not None and (not isinstance(port, int) or port < 1024 or port > 65535):
            issues.append(ConfigIssue("knowledge", "query_api_port", "error", "Port must be 1024-65535"))
        batch_time = kb.get("daily_batch_time")
        if batch_time is not None and not re.match(r"^\d{2}:\d{2}$", str(batch_time)):
            issues.append(ConfigIssue("knowledge", "daily_batch_time", "warning", "Expected HH:MM format"))
        return issues

    def validate_trust_section(self, trust: dict) -> list[ConfigIssue]:
        issues = []
        valid_levels = ("observe", "suggest", "draft", "execute_with_approval", "autonomous")
        level = trust.get("default_trust_level")
        if level is not None and level not in valid_levels:
            issues.append(ConfigIssue("trust", "default_trust_level", "error", f"Must be one of {valid_levels}"))
        for key in ("min_success_rate_for_suggestion",):
            val = trust.get(key)
            if val is not None and (not isinstance(val, (int, float)) or val < 0 or val > 1):
                issues.append(ConfigIssue("trust", key, "warning", f"{key} should be between 0 and 1"))
        obs = trust.get("min_observations_for_promotion")
        if obs is not None and (not isinstance(obs, int) or obs < 1):
            issues.append(ConfigIssue("trust", "min_observations_for_promotion", "warning", "Should be a positive integer"))
        return issues

    def validate_privacy_section(self, privacy: dict) -> list[ConfigIssue]:
        issues = []
        zones = privacy.get("zones", {})
        for key in ("auto_pause",):
            windows = zones.get(key, [])
            if isinstance(windows, list):
                for w in windows:
                    if not re.match(r"^\d{2}:\d{2}-\d{2}:\d{2}$", str(w)):
                        issues.append(ConfigIssue("privacy.zones", key, "warning", f"Invalid time window format: {w} (expected HH:MM-HH:MM)"))
        return issues

    def validate_features_section(self, features: dict) -> list[ConfigIssue]:
        issues = []
        valid_keys = ("activity_classification", "continuity_tracking", "lifecycle_management", "curation", "runtime_validation")
        for key in valid_keys:
            val = features.get(key)
            if val is not None and not isinstance(val, bool):
                issues.append(ConfigIssue("features", key, "warning", f"{key} should be true or false"))
        return issues

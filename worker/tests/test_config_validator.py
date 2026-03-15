from oc_apprentice_worker.config_validator import ConfigValidator, ConfigIssue

class TestValidConfig:
    def test_empty_config_no_issues(self):
        assert ConfigValidator().validate({}) == []

    def test_valid_vlm_section(self):
        config = {"vlm": {"annotation_model": "qwen3.5:2b", "max_jobs_per_day": 50}}
        assert ConfigValidator().validate(config) == []

class TestVlmValidation:
    def test_empty_model_name_warning(self):
        issues = ConfigValidator().validate({"vlm": {"annotation_model": ""}})
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_invalid_max_jobs(self):
        issues = ConfigValidator().validate({"vlm": {"max_jobs_per_day": -1}})
        assert len(issues) == 1

class TestKnowledgeValidation:
    def test_invalid_port(self):
        issues = ConfigValidator().validate({"knowledge": {"query_api_port": 80}})
        assert any(i.severity == "error" for i in issues)

    def test_valid_port(self):
        assert ConfigValidator().validate({"knowledge": {"query_api_port": 9477}}) == []

    def test_invalid_batch_time(self):
        issues = ConfigValidator().validate({"knowledge": {"daily_batch_time": "invalid"}})
        assert len(issues) == 1

class TestTrustValidation:
    def test_invalid_trust_level(self):
        issues = ConfigValidator().validate({"trust": {"default_trust_level": "god_mode"}})
        assert any(i.severity == "error" for i in issues)

    def test_invalid_threshold(self):
        issues = ConfigValidator().validate({"trust": {"min_success_rate_for_suggestion": 1.5}})
        assert len(issues) == 1

class TestPrivacyValidation:
    def test_invalid_time_window(self):
        issues = ConfigValidator().validate({"privacy": {"zones": {"auto_pause": ["invalid"]}}})
        assert len(issues) == 1

    def test_valid_time_window(self):
        assert ConfigValidator().validate({"privacy": {"zones": {"auto_pause": ["22:00-06:00"]}}}) == []

class TestFeaturesValidation:
    def test_non_bool_feature(self):
        issues = ConfigValidator().validate({"features": {"curation": "yes"}})
        assert len(issues) == 1

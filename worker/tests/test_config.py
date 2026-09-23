"""``config.load_config``: the worker's one reader of ``config.toml``."""

from __future__ import annotations

import platform
from datetime import time
from pathlib import Path

import pytest

from agenthandover_worker.config import (
    Config,
    FeaturesConfig,
    config_path,
    load_config,
)
from agenthandover_worker.scheduler import SchedulerConfig


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Location and whole-file behaviour
# ---------------------------------------------------------------------------


def test_default_location_follows_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    if platform.system() == "Darwin":
        expected = tmp_path / "Library" / "Application Support" / "agenthandover"
    else:
        expected = tmp_path / ".config" / "agenthandover"
    assert config_path() == expected / "config.toml"

    expected.mkdir(parents=True)
    (expected / "config.toml").write_text(
        "[vlm]\nannotation_timeout_seconds = 240\n[focus]\nmax_session_hours = 2.5\n"
    )
    cfg = load_config()
    assert cfg.path == expected / "config.toml"
    assert cfg.vlm.annotation_timeout_seconds == 240.0
    assert cfg.focus.max_session_hours == 2.5


def test_missing_file_gives_every_default(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg == Config()
    assert cfg.path is None
    assert cfg.vlm.annotation_enabled is True
    assert cfg.vlm.annotation_model == "qwen3.5:4b"
    assert cfg.vlm.sop_model == "qwen3.5:4b"
    assert cfg.vlm.annotation_max_frames_per_day == 1500
    assert cfg.vlm.annotation_max_compute_minutes_per_day == 60.0
    assert (cfg.embedding.retention_days, cfg.embedding.max_rows) == (90, 50_000)
    assert cfg.vlm.sliding_window_max_age_sec == 600
    assert cfg.vlm.ollama_host == "http://localhost:11434"
    assert cfg.vlm.annotation_timeout_seconds == 180.0
    assert cfg.vlm.mode == "local"
    assert cfg.focus.max_session_hours == 4.0
    assert cfg.sop.auto_approve is False
    assert cfg.knowledge.query_api_enabled is True
    assert cfg.knowledge.query_api_port == 9477
    assert cfg.embedding.image_embeddings is False
    assert cfg.export.skills_dir == Path.home() / ".claude" / "skills"
    assert cfg.privacy_zones == {} and cfg.raw == {}


def test_invalid_toml_gives_defaults(tmp_path, caplog):
    cfg = load_config(_write(tmp_path, "not valid toml {{{{"))
    assert cfg == Config()
    assert "using defaults" in caplog.text


def test_config_is_frozen(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    with pytest.raises(AttributeError):
        cfg.vlm.ollama_host = "http://elsewhere"  # type: ignore[misc]


def test_raw_and_privacy_zones_are_passed_through(tmp_path):
    cfg = load_config(_write(tmp_path, (
        '[privacy.zones]\nblocked = ["com.1password.*"]\n'
        '[storage]\nretention_days_raw = 7\n'
    )))
    assert cfg.privacy_zones == {"blocked": ["com.1password.*"]}
    assert cfg.raw["storage"]["retention_days_raw"] == 7


# ---------------------------------------------------------------------------
# [vlm]
# ---------------------------------------------------------------------------


def test_vlm_values(tmp_path):
    cfg = load_config(_write(tmp_path, (
        '[vlm]\nannotation_enabled = false\nannotation_model = "qwen3:4b"\n'
        'sop_model = "gemma4:e4b"\nannotation_max_frames_per_day = 5\n'
        'sliding_window_max_age_sec = 900\nollama_host = "http://127.0.0.1:1"\n'
        'mode = "remote"\nprovider = "openai"\nmodel = "gpt"\n'
        'api_key_env = "KEY"\n'
    )))
    vlm = cfg.vlm
    assert vlm.annotation_enabled is False
    assert vlm.annotation_model == "qwen3:4b"
    assert vlm.sop_model == "gemma4:e4b"
    assert vlm.annotation_max_frames_per_day == 5
    assert vlm.sliding_window_max_age_sec == 900
    assert vlm.ollama_host == "http://127.0.0.1:1"
    assert (vlm.mode, vlm.provider, vlm.model, vlm.api_key_env) == (
        "remote", "openai", "gpt", "KEY",
    )


@pytest.mark.parametrize("raw, expected", [
    ('"false"', False), ('"yes"', True), ("0", False), ("true", True),
])
def test_booleans_written_as_strings_or_numbers(tmp_path, raw, expected):
    cfg = load_config(_write(tmp_path, f"[vlm]\nannotation_enabled = {raw}\n"))
    assert cfg.vlm.annotation_enabled is expected


def test_invalid_values_keep_the_default(tmp_path, caplog):
    cfg = load_config(_write(tmp_path, (
        '[vlm]\nannotation_timeout_seconds = "soon"\nannotation_max_frames_per_day = 0\n'
        'ollama_host = 11434\nannotation_enabled = "maybe"\n'
        "[focus]\nmax_session_hours = -1\n"
    )))
    assert cfg.vlm.annotation_timeout_seconds == 180.0
    assert cfg.vlm.annotation_max_frames_per_day == 1500
    assert cfg.vlm.ollama_host == "http://localhost:11434"
    assert cfg.vlm.annotation_enabled is True
    assert cfg.focus.max_session_hours == 4.0
    assert "[focus] max_session_hours = -1 is invalid" in caplog.text


def test_a_section_that_is_not_a_table(tmp_path, caplog):
    cfg = load_config(_write(tmp_path, 'vlm = "gemma"\n'))
    assert cfg.vlm == Config().vlm
    assert "[vlm]" in caplog.text


# ---------------------------------------------------------------------------
# [features] (was main._read_feature_flags)
# ---------------------------------------------------------------------------


FEATURES = (
    "activity_classification", "continuity_tracking", "lifecycle_management",
    "curation", "runtime_validation",
)


def test_features_default_on_and_complete(tmp_path):
    features = load_config(_write(tmp_path, "[vlm]\nenabled = true\n")).features
    assert set(vars(features)) == set(FEATURES)
    assert all(vars(features).values())


def test_single_feature_disabled(tmp_path):
    features = load_config(_write(tmp_path, "[features]\ncuration = false\n")).features
    assert features.curation is False
    assert features.activity_classification is True


def test_all_features_disabled_and_unknown_keys_ignored(tmp_path):
    text = "[features]\nfuture_feature = true\n" + "".join(
        f"{name} = false\n" for name in FEATURES
    )
    features = load_config(_write(tmp_path, text)).features
    assert not any(vars(features).values())
    assert not hasattr(features, "future_feature")
    assert FeaturesConfig().curation is True


def test_non_bool_feature_coerced(tmp_path):
    assert load_config(_write(tmp_path, "[features]\ncuration = 0\n")).features.curation is False


# ---------------------------------------------------------------------------
# [export] claude_skills_dir (was main._skills_dir)
# ---------------------------------------------------------------------------


def test_skills_dir_from_config(tmp_path):
    target = tmp_path / "skills-out"
    cfg = load_config(_write(tmp_path, f'[export]\nclaude_skills_dir = "{target}"\n'))
    assert cfg.export.skills_dir == target


def test_skills_dir_expands_tilde(tmp_path):
    cfg = load_config(_write(tmp_path, '[export]\nclaude_skills_dir = "~/agent-skills"\n'))
    assert cfg.export.skills_dir == Path.home() / "agent-skills"


def test_skills_dir_missing_or_invalid_uses_default(tmp_path):
    default = Path.home() / ".claude" / "skills"
    assert load_config(_write(tmp_path, '[export]\nadapter = "openclaw"\n')).export.skills_dir == default
    assert load_config(_write(tmp_path, "[export]\nclaude_skills_dir = 3\n")).export.skills_dir == default


# ---------------------------------------------------------------------------
# [llm], [sop], [knowledge], [embedding], [openclaw]
# ---------------------------------------------------------------------------


def test_export_adapter_defaults_to_openclaw(tmp_path):
    assert load_config(tmp_path / "missing.toml").export.adapter == "openclaw"
    cfg = load_config(_write(tmp_path, '[export]\njson_export = false\n'))
    assert cfg.export.adapter == "openclaw"


def test_export_adapter_from_config(tmp_path):
    cfg = load_config(_write(
        tmp_path, '[export]\nadapter = "all"\nclaude_skills_dir = "~/s"\n',
    ))
    assert cfg.export.adapter == "all"
    assert cfg.export.skills_dir == Path.home() / "s"


def test_unknown_export_adapter_keeps_the_default(tmp_path, caplog):
    cfg = load_config(_write(tmp_path, '[export]\nadapter = "cursor"\n'))
    assert cfg.export.adapter == "openclaw"
    assert "adapter" in caplog.text


def test_worker_adapter_flag_falls_back_to_config():
    """``--adapter`` absent: main() takes ``[export] adapter`` (a pkg install writes "all")."""
    from agenthandover_worker.main import _parse_args

    assert _parse_args([]).adapter is None
    assert _parse_args(["--adapter", "generic"]).adapter == "generic"


def test_pkg_default_config_exports_through_every_adapter(tmp_path):
    """The config.toml the pkg postinstall writes selects ``adapter = "all"``."""
    script = (
        Path(__file__).resolve().parents[2] / "resources" / "pkg" / "scripts" / "postinstall"
    ).read_text(encoding="utf-8")
    body = script.split("<< 'TOML'\n", 1)[1].split("\nTOML\n", 1)[0]
    assert load_config(_write(tmp_path, body)).export.adapter == "all"


def test_other_sections(tmp_path):
    cfg = load_config(_write(tmp_path, (
        '[llm]\nenhance_sops = false\nmax_enhancements_per_day = 5\nmodel = "m"\n'
        "timeout_seconds = 30\ntemperature = 1\nmax_tokens = 100\n"
        "[sop]\nauto_approve = true\n"
        "[knowledge]\nquery_api_enabled = false\nquery_api_port = 9999\n"
        "[embedding]\nimage_embeddings = true\n"
        '[openclaw]\nworkspace_path = "~/ws"\n'
    )))
    assert cfg.llm.enhance_sops is False
    assert cfg.llm.max_enhancements_per_day == 5
    assert cfg.llm.model == "m"
    assert cfg.llm.timeout_seconds == 30
    assert cfg.llm.temperature == 1.0 and isinstance(cfg.llm.temperature, float)
    assert cfg.llm.max_tokens == 100
    assert cfg.sop.auto_approve is True
    assert cfg.knowledge.query_api_enabled is False
    assert cfg.knowledge.query_api_port == 9999
    assert cfg.embedding.image_embeddings is True
    assert cfg.openclaw.workspace_path == "~/ws"


# ---------------------------------------------------------------------------
# [idle_jobs] (was main._read_idle_jobs_config)
# ---------------------------------------------------------------------------


def test_idle_jobs_defaults_match_the_scheduler(tmp_path):
    kwargs = load_config(tmp_path / "nope.toml").idle_jobs.scheduler_kwargs()
    assert SchedulerConfig(**kwargs) == SchedulerConfig()


def test_idle_jobs_values_and_window(tmp_path):
    cfg = load_config(_write(tmp_path, (
        "[idle_jobs]\nrequire_ac_power = false\nmin_battery_percent = 20\n"
        'max_cpu_percent = 60\nmax_temp_c = 90\nrun_window_local_time = "22:30-06:15"\n'
    )))
    scheduler = SchedulerConfig(**cfg.idle_jobs.scheduler_kwargs())
    assert scheduler.require_ac_power is False
    assert scheduler.min_battery_percent == 20
    assert scheduler.max_cpu_percent == 60
    assert scheduler.max_temp_c == 90
    assert (scheduler.run_window_start, scheduler.run_window_end) == (
        time(22, 30), time(6, 15),
    )


def test_idle_jobs_bad_window_keeps_the_default(tmp_path, caplog):
    cfg = load_config(_write(tmp_path, (
        '[idle_jobs]\nmax_temp_c = 70\nrun_window_local_time = "late"\n'
    )))
    assert cfg.idle_jobs.max_temp_c == 70
    assert (cfg.idle_jobs.run_window_start, cfg.idle_jobs.run_window_end) == (
        time(1, 0), time(5, 0),
    )
    assert "run_window_local_time" in caplog.text

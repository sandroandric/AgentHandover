"""Cloud mode was removed in 0.5.0 (WP-F13).

Every model runs locally through Ollama.  An existing ``[vlm]`` block in
cloud mode (``mode = "remote"`` with ``provider``, ``model`` and
``api_key_env``, as the app's onboarding or ``agenthandover-setup-vlm
--remote`` wrote it) is made local by the upgrade's migration
(``model_profiles.migrate_vlm_block``: the installer's ``migrate-vlm`` and
the worker at startup), with a backup and one log line; a local block,
customised or not, is left to the tier migration.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from agenthandover_worker import main as worker_main
from agenthandover_worker import model_profiles as mp
from agenthandover_worker import sop_enhancer
from agenthandover_worker.config import VlmConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SRC = REPO_ROOT / "worker" / "src" / "agenthandover_worker"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "vlm_block"
NOW = datetime(2026, 9, 24, 9, 30, 0)

# What 0.4.x onboarding left: the installer's tier block, then the cloud
# choice written over its managed keys (ModelTierTable.remoteVLMBlock).
REMOTE = """\
# my config
[export]
adapter = "all"

[vlm]
mode = "remote"
provider = "anthropic"
model = "claude-sonnet-4-6-20260320"
api_key_env = "ANTHROPIC_API_KEY"
tier = "performance"
annotation_model = "qwen3.5:4b"
sop_model = "gemma4:12b-it-qat"
annotation_timeout_seconds = 240

[embedding]
model = "nomic-embed-text"

[privacy.zones]
# 1Password, any version
blocked = ["1password*"]
"""

LOCAL = REMOTE.replace(
    'mode = "remote"\nprovider = "anthropic"\nmodel = "claude-sonnet-4-6-20260320"\n'
    'api_key_env = "ANTHROPIC_API_KEY"\n',
    'mode = "local"\n',
)


# --- the migration -----------------------------------------------------------


def test_a_remote_block_becomes_local_keeping_its_models_and_every_other_line():
    new, changed, reason = mp.migrate_vlm_block(REMOTE)
    assert changed
    assert new == LOCAL
    assert reason == (
        'cloud mode was removed in 0.5.0: mode "remote" (provider anthropic, '
        'model claude-sonnet-4-6-20260320) -> "local"; the local models keep running as before'
    )
    vlm = tomllib.loads(new)["vlm"]
    assert vlm["mode"] == "local"
    assert not {"provider", "model", "api_key_env"} & set(vlm)
    # The embedding model is [embedding]'s, not cloud mode's.
    assert tomllib.loads(new)["embedding"]["model"] == "nomic-embed-text"


def test_the_migration_is_idempotent():
    once, changed, _ = mp.migrate_vlm_block(REMOTE)
    assert changed
    again, changed, reason = mp.migrate_vlm_block(once)
    assert (again, changed) == (once, False)
    assert reason == 'already the current models of tier "performance"'


def test_a_remote_block_on_a_tiers_earlier_models_gets_both_migrations():
    old = FIXTURES.joinpath("recommended-previous.toml").read_text(encoding="utf-8")
    migrated = FIXTURES.joinpath("recommended-migrated.toml").read_text(encoding="utf-8")
    remote = old.replace(
        'mode = "local"\n',
        'mode = "remote"\nprovider = "openai"\nmodel = "gpt-4.1-mini"\napi_key_env = "OPENAI_API_KEY"\n',
    )
    new, changed, reason = mp.migrate_vlm_block(remote)
    assert changed
    assert new == migrated
    assert reason.startswith('cloud mode was removed in 0.5.0: mode "remote" (provider openai')
    assert '; tier "recommended": sop_model gemma4:12b-it-qat -> qwen3.5:4b' in reason


def test_a_remote_block_with_only_the_cloud_keys():
    # setup-vlm --remote on a config without a [vlm] section appended this.
    new, changed, _ = mp.migrate_vlm_block(
        '[export]\nadapter = "all"\n\n[vlm]\nmode = "remote"\nprovider = "google"\n'
        'model = "gemini-2.5-flash"\napi_key_env = "GOOGLE_API_KEY"\n'
    )
    assert changed
    assert new == '[export]\nadapter = "all"\n\n[vlm]\nmode = "local"\n'
    # The loader fills the models from this Mac's RAM tier, as it did.
    assert new == mp.migrate_vlm_block(new)[0]


def test_hand_written_cloud_lines_and_crlf():
    text = (
        "[vlm]\r\n"
        "mode = 'remote'   # cloud\r\n"
        "provider='openai'\r\n"
        "  model = \"gpt-4.1-mini\"  # the cheap one\r\n"
        "tier = \"standard\"\r\n"
        "[focus]\r\nmax_session_hours = 2\r\n"
    )
    new, changed, _ = mp.migrate_vlm_block(text)
    assert changed
    assert new == '[vlm]\r\nmode = "local"\r\ntier = "standard"\r\n[focus]\r\nmax_session_hours = 2\r\n'


@pytest.mark.parametrize(("label", "text", "reason"), [
    # A multi-line value: dropping its first line would break the file.
    ("multi-line cloud value",
     '[vlm]\nmode = "remote"\nmodel = """\ngpt-4.1-mini\n"""\ntier = "standard"\n',
     '[vlm] mode is "remote"'),
    # A header the line scan does not recognise.
    ("header with a comment",
     '[vlm]  # models\nmode = "remote"\nprovider = "openai"\n',
     "no [vlm] section"),
])
def test_a_block_it_cannot_rewrite_safely_is_left_for_doctor(label, text, reason):
    assert mp.migrate_vlm_block(text) == (text, False, reason), label


@pytest.mark.parametrize("text", [
    LOCAL,
    LOCAL.replace('mode = "local"\n', ""),
    # A customised local block that still carries a leftover provider line:
    # not cloud mode, not touched.
    LOCAL.replace('mode = "local"\n', 'mode = "local"\nprovider = "openai"\n'),
    FIXTURES.joinpath("input.toml").read_text(encoding="utf-8"),
    "",
])
def test_local_blocks_are_not_touched_by_the_cloud_migration(text):
    assert mp.remote_to_local(text) == (text, None)


def test_the_file_is_backed_up_then_rewritten(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(REMOTE, encoding="utf-8")
    cfg.chmod(0o600)
    changed, reason, backup = mp.migrate_vlm_file(cfg, now=NOW)
    assert changed and reason.startswith("cloud mode was removed in 0.5.0")
    assert backup == tmp_path / "config.toml.20260924-093000.bak"
    assert backup.read_text(encoding="utf-8") == REMOTE
    assert cfg.read_text(encoding="utf-8") == LOCAL
    assert oct(cfg.stat().st_mode & 0o777) == oct(0o600)
    assert mp.migrate_vlm_file(cfg, now=NOW) == (
        False, 'already the current models of tier "performance"', None,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "config.toml", "config.toml.20260924-093000.bak",
    ]


def test_the_installers_migrate_vlm_command(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(REMOTE, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "agenthandover_worker.model_profiles", "migrate-vlm", str(cfg)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith(f"Migrated [vlm] in {cfg}: cloud mode was removed in 0.5.0")
    assert cfg.read_text(encoding="utf-8") == LOCAL


def test_worker_startup_migrates_with_one_info_line(tmp_path, caplog):
    cfg = tmp_path / "config.toml"
    cfg.write_text(REMOTE, encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="agenthandover_worker"):
        assert worker_main._migrate_vlm_config(cfg) is True
    lines = [r for r in caplog.records if "[vlm]" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "cloud mode was removed in 0.5.0" in lines[0].getMessage()
    assert cfg.read_text(encoding="utf-8") == LOCAL


# --- nothing reads cloud mode any more ----------------------------------------


def test_the_loader_has_no_cloud_fields_and_ignores_an_unmigrated_block(
    tmp_path, monkeypatch,
):
    assert not {"mode", "provider", "model", "api_key_env"} & set(VlmConfig.__dataclass_fields__)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[vlm]\nmode = "remote"\nprovider = "openai"\nmodel = "gpt-4.1-mini"\n'
        'tier = "standard"\nollama_host = "http://127.0.0.1:9"\n',
        encoding="utf-8",
    )
    vlm = load_config(cfg).vlm
    standard = mp.get_tier("standard")
    assert (vlm.annotation_model, vlm.sop_model) == (standard.annotation_model, standard.sop_model)


def test_the_cloud_backends_and_setup_vlm_are_gone():
    for module in (
        "agenthandover_worker.backends",
        "agenthandover_worker.vlm_worker",
        "agenthandover_worker.setup_vlm",
    ):
        assert importlib.util.find_spec(module) is None, module
    for path in WORKER_SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("import anthropic", "import openai", "google.genai", "import ollama",
                       "find-generic-password"):
            assert needle not in text, f"{path.name}: {needle}"
        # Only the upgrade's migration still names cloud mode's key.
        if path.name != "model_profiles.py":
            assert "api_key_env" not in text, path.name
    pyproject = tomllib.loads((REPO_ROOT / "worker" / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert not {"vlm-openai", "vlm-anthropic", "vlm-google", "vlm-remote", "vlm-all"} & set(extras)
    assert "agenthandover-setup-vlm" not in pyproject["project"]["scripts"]


def test_worker_status_has_no_cloud_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_main, "_status_dir", lambda: tmp_path)
    worker_main._write_worker_status(
        started_at="2026-09-24T09:00:00Z", events_processed_today=0, sops_generated=0,
        last_pipeline_duration_ms=None, consecutive_errors=0, vlm_available=True,
    )
    text = (tmp_path / "worker-status.json").read_text(encoding="utf-8")
    assert '"vlm_available": true' in text
    assert "vlm_mode" not in text and "vlm_provider" not in text


# --- vlm_available and the SOP enhancer: the local Ollama ---------------------


class _FakeClient:
    """Stands in for ``OllamaClient`` (records hosts and calls)."""

    calls: list[tuple] = []
    served: set[str] = set()

    def __init__(self, host: str) -> None:
        self.host = host

    def has_model(self, model: str, timeout: float | None = 3.0) -> bool:
        _FakeClient.calls.append(("has_model", self.host, model))
        return model in _FakeClient.served

    def generate(self, model, prompt, **kwargs):
        _FakeClient.calls.append(("generate", self.host, model, kwargs))

        class _R:
            text = '{"task_description": "Sends the report.", "execution_overview": {}}'

        return _R()


@pytest.fixture()
def fake_ollama(monkeypatch):
    from agenthandover_worker import ollama

    _FakeClient.calls = []
    _FakeClient.served = set()
    monkeypatch.setattr(ollama, "OllamaClient", _FakeClient)
    monkeypatch.setattr(sop_enhancer, "OllamaClient", _FakeClient)
    return _FakeClient


def test_vlm_available_is_the_annotation_model_at_the_configured_host(fake_ollama):
    vlm = VlmConfig(ollama_host="http://gpu-box.lan:11434", annotation_model="qwen3.5:4b")
    assert worker_main._vlm_backend_available(vlm) is False
    fake_ollama.served = {"qwen3.5:4b"}
    assert worker_main._vlm_backend_available(vlm) is True
    assert fake_ollama.calls == [
        ("has_model", "http://gpu-box.lan:11434", "qwen3.5:4b"),
        ("has_model", "http://gpu-box.lan:11434", "qwen3.5:4b"),
    ]


def test_the_enhancer_runs_the_sop_model_on_the_configured_ollama(fake_ollama):
    backend = sop_enhancer.create_llm_backend(
        {"model": "", "timeout_seconds": 30, "temperature": 0.2, "max_tokens": 700},
        ollama_host="http://gpu-box.lan:11434", sop_model="qwen3.5:9b",
    )
    assert backend.model == "qwen3.5:9b"
    assert fake_ollama.calls == [], "no model call until an enhancement"
    assert backend.is_available() is False
    fake_ollama.served = {"qwen3.5:9b"}
    assert backend.is_available() is False, "cached for a minute"
    backend._checked_at -= 61
    assert backend.is_available() is True

    enhancer = sop_enhancer.SOPEnhancer(backend)
    out = enhancer.enhance_sop({"slug": "s", "title": "Report", "steps": [{"step": "send"}]})
    assert out["task_description"] == "Sends the report."
    (_, host, model, kwargs), = [c for c in fake_ollama.calls if c[0] == "generate"]
    assert (host, model) == ("http://gpu-box.lan:11434", "qwen3.5:9b")
    assert kwargs["format"] == "json" and kwargs["think"] is False
    assert kwargs["options"] == {"temperature": 0.2, "num_predict": 700}
    assert kwargs["timeout"] == 30.0 and kwargs["call_type"] == "sop"
    assert "workflow documentation expert" in kwargs["system"]


def test_the_enhancer_model_override_and_no_model():
    assert sop_enhancer.create_llm_backend({"model": "llama3.2:3b"}, sop_model="x").model == "llama3.2:3b"
    assert sop_enhancer.create_llm_backend({"model": ""}, sop_model="") is None

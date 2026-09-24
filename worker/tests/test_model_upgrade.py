"""Upgrading a real 0.4.0 config to 0.5's models (WP-F16).

v0.4.0's onboarding wrote ``[vlm]`` with ``annotation_model`` and
``sop_model`` only (no ``tier``, no ``mode``);
``tests/fixtures/vlm_block/v0.4.0-onboarding.toml`` is the file its
installer and onboarding left on a Mac that picked Recommended or
Performance, byte for byte.  0.5 recognises those pairs, but switches only
to models Ollama serves: until then the Mac keeps the models it has, the
app offers the download (``upgrade-plan``, then ``upgrade-apply``), and
the worker makes the switch at startup once they are pulled.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agenthandover_worker import main as worker_main
from agenthandover_worker import model_profiles as mp

REPO_ROOT = Path(__file__).resolve().parents[2]
V040 = (REPO_ROOT / "tests/fixtures/vlm_block/v0.4.0-onboarding.toml").read_text(encoding="utf-8")
GEMMA_12B = ("gemma4:12b-it-qat", "gemma4:12b-it-qat")
V040_BLOCK = '\n[vlm]\nannotation_model = "gemma4:12b-it-qat"\nsop_model = "gemma4:12b-it-qat"\n'
SERVED = frozenset(m for t in mp.TIERS for m in t.models)


def v040(annotation_model: str, sop_model: str) -> str:
    """The 0.4.0 file of the tier that wrote this pair."""
    return V040.replace(
        V040_BLOCK,
        f'\n[vlm]\nannotation_model = "{annotation_model}"\nsop_model = "{sop_model}"\n',
    )


# v0.4.0's cloud setup (``writeRemoteVLMConfig``) on the installer's file:
# a [vlm] block with the four cloud keys and no models.
V040_TEMPLATE = V040.split("\n\n[embedding]")[0].removesuffix("\n") + "\n\n[embedding]\n" \
    'model = "nomic-embed-text"\nimage_embeddings = true\n'
CLOUD_FIELDS = (
    'mode = "remote"\nprovider = "openai"\nmodel = "gpt-4.1-mini"\n'
    'api_key_env = "OPENAI_API_KEY"\n'
)
V040_CLOUD = V040_TEMPLATE + "\n[vlm]\n" + CLOUD_FIELDS
# Local models first, then cloud: the cloud keys went after them.
V040_CLOUD_AFTER_LOCAL = V040 + "\n" + CLOUD_FIELDS


def test_the_fixture_is_what_v040_wrote():
    parsed = tomllib.loads(V040)
    assert parsed["vlm"] == {
        "annotation_model": "gemma4:12b-it-qat", "sop_model": "gemma4:12b-it-qat",
    }
    assert parsed["export"]["adapter"] == "openclaw"
    assert V040.endswith(V040_BLOCK)


# --- the pairs and the tier each means --------------------------------------


@pytest.mark.parametrize(("pair", "ram", "tier_id", "target", "models"), [
    # 0.4.0 Standard.
    (("qwen3.5:2b", "qwen3.5:4b"), 8, "standard",
     ("qwen3.5:2b-q4_K_M", "qwen3.5:4b"), ["qwen3.5:2b-q4_K_M", "qwen3.5:4b"]),
    # 0.4.0 Recommended / Performance: Gemma 4 12B for both, by RAM.
    (GEMMA_12B, 16, "recommended", ("qwen3.5:4b", "qwen3.5:4b"), ["qwen3.5:4b"]),
    (GEMMA_12B, 24, "performance",
     ("qwen3.5:4b", "gemma4:12b-it-qat"), ["qwen3.5:4b", "gemma4:12b-it-qat"]),
    # 0.4.0 Max Quality.
    (("gemma4:31b-it-qat", "gemma4:31b-it-qat"), 64, "max_quality",
     ("qwen3.5:4b", "gemma4:31b-it-qat"), ["qwen3.5:4b", "gemma4:31b-it-qat"]),
])
def test_plan_for_each_v040_pair(pair, ram, tier_id, target, models):
    text = v040(*pair)
    plan, reason = mp.upgrade_plan(text, ram_gb=ram)
    assert plan == {
        "tier": tier_id,
        "label": mp.get_tier(tier_id).label,
        "current": {"annotation_model": pair[0], "sop_model": pair[1]},
        "target": {"annotation_model": target[0], "sop_model": target[1]},
        "models": models,
        "sizes_gb": {m: mp.MODEL_SIZES_GB[m] for m in models},
    }
    assert f"{pair[0]} + {pair[1]} was 0.4.0's default" in reason
    # Read only.
    assert text == v040(*pair)


@pytest.mark.parametrize("ram", [0, 8, 12, 16, 23, 24, 32, 48, 64, 128])
def test_the_gemma_12b_pair_picks_by_ram_as_onboarding_recommends(ram):
    """16 GB and 24 GB wrote the same pair; 0.5's tier table tells them
    apart by the same ``min_ram_gb`` onboarding recommends with.  A Mac
    past 24 GB stays on Performance (its user chose Gemma 12B over 0.4.0's
    Max Quality), one under 16 GB gets Recommended (the smallest of the two)."""
    tier = mp.untiered_default_tier(GEMMA_12B, ram)
    recommended = mp.recommend_tier(ram)
    if recommended.id in ("recommended", "performance"):
        assert tier == recommended
    elif recommended.min_ram_gb > mp.get_tier("performance").min_ram_gb:
        assert tier.id == "performance"
    else:
        assert tier.id == "recommended"


def test_tiers_json_carries_the_untiered_defaults():
    assert mp.tiers_document()["untiered_defaults"] == [
        {"annotation_model": "qwen3.5:2b", "sop_model": "qwen3.5:4b", "tiers": ["standard"]},
        {"annotation_model": "gemma4:12b-it-qat", "sop_model": "gemma4:12b-it-qat",
         "tiers": ["recommended", "performance"]},
        {"annotation_model": "gemma4:31b-it-qat", "sop_model": "gemma4:31b-it-qat",
         "tiers": ["max_quality"]},
    ]
    for (ann, sop), ids in mp.UNTIERED_DEFAULTS:
        assert all(mp.get_tier(i) for i in ids)
        # Never a pair some 0.5 tier runs: that would be no upgrade.
        assert all((t.annotation_model, t.sop_model) != (ann, sop) for t in mp.TIERS)


# --- the switch: only to models Ollama serves -------------------------------


def test_switch_when_the_models_are_served():
    new, changed, reason = mp.migrate_vlm_block(V040, served=["qwen3.5:4b"], ram_gb=16)
    assert changed
    assert new == V040.replace(V040_BLOCK, "\n" + mp.vlm_block(mp.get_tier("recommended")))
    assert new == mp.apply_vlm_block(V040, mp.get_tier("recommended"))
    assert reason == (
        'tier "recommended": annotation_model gemma4:12b-it-qat -> qwen3.5:4b, '
        "sop_model gemma4:12b-it-qat -> qwen3.5:4b (gemma4:12b-it-qat + gemma4:12b-it-qat "
        "was 0.4.0's default; it now runs qwen3.5:4b + qwen3.5:4b)"
    )
    before, after = tomllib.loads(V040), tomllib.loads(new)
    assert after["vlm"] == {
        "mode": "local", "tier": "recommended",
        "annotation_model": "qwen3.5:4b", "sop_model": "qwen3.5:4b",
    }
    assert {k: v for k, v in after.items() if k != "vlm"} == {
        k: v for k, v in before.items() if k != "vlm"
    }


@pytest.mark.parametrize(("served", "waits_for"), [
    (None, "the switch waits until Ollama serves qwen3.5:4b, gemma4:12b-it-qat "),
    ([], "the switch waits until qwen3.5:4b, gemma4:12b-it-qat (~10.6 GB) are downloaded"),
    (["gemma4:12b-it-qat", "nomic-embed-text:latest"],
     "the switch waits until qwen3.5:4b (~3.4 GB) is downloaded"),
])
def test_keep_the_models_until_the_new_ones_are_pulled(served, waits_for):
    new, changed, reason = mp.migrate_vlm_block(V040, served=served, ram_gb=24)
    assert (new, changed) == (V040, False)
    assert waits_for in reason
    assert reason.endswith("agenthandover setup --tier performance)")


def test_the_switch_is_idempotent():
    once, changed, _ = mp.migrate_vlm_block(V040, served=SERVED, ram_gb=24)
    assert changed
    again, changed, reason = mp.migrate_vlm_block(once, served=SERVED, ram_gb=24)
    assert (again, changed) == (once, False)
    assert reason == 'already the current models of tier "performance"'
    assert mp.upgrade_plan(once, ram_gb=24)[0] is None
    # Another Mac's RAM does not move a block that names its tier.
    assert mp.upgrade_plan(once, ram_gb=16)[0] is None


@pytest.mark.parametrize(("label", "text", "reason"), [
    ("another pair",
     v040("qwen3.5:4b", "gemma4:31b-it-qat"),
     "[vlm] names no tier and qwen3.5:4b + gemma4:31b-it-qat is not a default of 0.4.0 "
     "(set by hand)"),
    ("the pair swapped",
     v040("gemma4:31b-it-qat", "qwen3.5:4b"),
     "[vlm] names no tier and gemma4:31b-it-qat + qwen3.5:4b is not a default of 0.4.0 "
     "(set by hand)"),
    ("a comment on a model line",
     V040.replace('sop_model = "gemma4:12b-it-qat"', 'sop_model = "gemma4:12b-it-qat"  # mine'),
     '[vlm] sop_model was edited by hand (sop_model = "gemma4:12b-it-qat"  # mine)'),
    ("only one model pinned",
     V040.replace('sop_model = "gemma4:12b-it-qat"\n', ""),
     "[vlm] names no tier"),
])
def test_a_hand_customised_block_is_left_alone(label, text, reason):
    assert mp.upgrade_plan(text, ram_gb=24) == (None, reason), label
    assert mp.migrate_vlm_block(text, served=SERVED, ram_gb=24) == (text, False, reason)


def test_a_hand_set_key_in_the_block_survives_the_switch():
    text = V040.replace(
        'sop_model = "gemma4:12b-it-qat"\n',
        'sop_model = "gemma4:12b-it-qat"\nannotation_timeout_seconds = 240\n',
    )
    new, changed, _ = mp.migrate_vlm_block(text, served=SERVED, ram_gb=16)
    assert changed
    assert new.endswith(mp.vlm_block(mp.get_tier("recommended")) + "annotation_timeout_seconds = 240\n")


# --- 0.4.0's cloud setup -----------------------------------------------------


def test_a_v040_cloud_config_becomes_local_and_has_no_switch():
    new, changed, reason = mp.migrate_vlm_block(V040_CLOUD)
    assert changed
    assert new == V040_TEMPLATE + '\n[vlm]\nmode = "local"\n'
    assert reason.startswith('cloud mode was removed in 0.5.0: mode "remote" (provider openai')
    assert mp.upgrade_plan(V040_CLOUD) == (None, "[vlm] names no tier")


def test_a_v040_cloud_config_with_local_models_waits_for_them_too():
    plan, _ = mp.upgrade_plan(V040_CLOUD_AFTER_LOCAL, ram_gb=16)
    assert plan is not None and plan["tier"] == "recommended"
    # The installer: cloud to local only, the models stay.
    new, changed, reason = mp.migrate_vlm_block(V040_CLOUD_AFTER_LOCAL)
    assert changed
    assert new == V040.replace("[vlm]\n", '[vlm]\nmode = "local"\n') + "\n"
    assert "cloud mode was removed" in reason and "the switch waits until" in reason
    # The worker, once qwen3.5:4b is pulled: both.
    both, changed, _ = mp.migrate_vlm_block(V040_CLOUD_AFTER_LOCAL, served=SERVED, ram_gb=16)
    assert changed
    assert tomllib.loads(both)["vlm"] == {
        "mode": "local", "tier": "recommended",
        "annotation_model": "qwen3.5:4b", "sop_model": "qwen3.5:4b",
    }


# --- a fake Ollama and the commands ------------------------------------------


class _FakeOllama:
    def __init__(self) -> None:
        self.models: list[str] = []
        self.requests: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (http.server's name)
                fake.requests.append(self.path)
                body = json.dumps({
                    "models": [{"name": m, "model": m} for m in fake.models],
                }).encode()
                self.send_response(200 if self.path == "/api/tags" else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture()
def ollama():
    fake = _FakeOllama()
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


def _config(tmp_path: Path, host: str, text: str = V040) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        text.replace(V040_BLOCK, V040_BLOCK + f'ollama_host = "{host}"\n'), encoding="utf-8",
    )
    return cfg


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agenthandover_worker.model_profiles", *args],
        capture_output=True, text=True, check=False,
    )


def test_upgrade_plan_command_prints_the_plan(tmp_path):
    cfg = _config(tmp_path, "http://127.0.0.1:9")
    proc = _run("upgrade-plan", str(cfg))
    assert proc.returncode == 0, proc.stderr
    plan, reason = mp.upgrade_plan(cfg.read_text(encoding="utf-8"))
    assert json.loads(proc.stdout) == {"plan": plan, "reason": reason}
    assert plan is not None and plan["current"]["sop_model"] == "gemma4:12b-it-qat"
    missing = _run("upgrade-plan", str(tmp_path / "none.toml"))
    assert missing.returncode == 0
    assert json.loads(missing.stdout) == {"plan": None, "reason": f"no {tmp_path / 'none.toml'}"}


def test_upgrade_apply_refuses_until_ollama_serves_the_models(tmp_path, ollama):
    cfg = _config(tmp_path, ollama.host)
    before = cfg.read_text(encoding="utf-8")
    plan, _ = mp.upgrade_plan(before)

    ollama.models = ["gemma4:12b-it-qat", "nomic-embed-text:latest"]
    proc = _run("upgrade-apply", str(cfg))
    assert proc.returncode == mp.EXIT_MODELS_MISSING
    assert proc.stderr == f"Kept [vlm] in {cfg}: {', '.join(plan['models'][:1])} not downloaded\n"
    assert cfg.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("config.toml.*.bak")) == []

    ollama.models = sorted(SERVED)
    proc = _run("upgrade-apply", str(cfg))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith(f'Migrated [vlm] in {cfg}: tier "{plan["tier"]}": ')
    after = tomllib.loads(cfg.read_text(encoding="utf-8"))["vlm"]
    assert (after["annotation_model"], after["sop_model"]) == (
        plan["target"]["annotation_model"], plan["target"]["sop_model"],
    )
    assert after["ollama_host"] == ollama.host
    [backup] = tmp_path.glob("config.toml.*.bak")
    assert backup.read_text(encoding="utf-8") == before

    again = _run("upgrade-apply", str(cfg))
    assert again.returncode == 0
    assert again.stdout.startswith(f"No model switch for [vlm] in {cfg}: already the current")


def test_upgrade_apply_refuses_when_ollama_does_not_answer(tmp_path):
    cfg = _config(tmp_path, "http://127.0.0.1:9")
    before = cfg.read_text(encoding="utf-8")
    proc = _run("upgrade-apply", str(cfg))
    assert proc.returncode == mp.EXIT_MODELS_MISSING
    assert proc.stderr == f"Kept [vlm] in {cfg}: Ollama does not answer at http://127.0.0.1:9\n"
    assert cfg.read_text(encoding="utf-8") == before


# --- worker startup ------------------------------------------------------------


def test_worker_startup_keeps_the_models_while_one_is_missing(tmp_path, ollama, caplog):
    cfg = _config(tmp_path, ollama.host)
    before = cfg.read_text(encoding="utf-8")
    ollama.models = ["gemma4:12b-it-qat", "nomic-embed-text:latest"]
    with caplog.at_level(logging.INFO, logger="agenthandover_worker"):
        assert worker_main._migrate_vlm_config(cfg) is False
    assert cfg.read_text(encoding="utf-8") == before
    lines = [r for r in caplog.records if "[vlm]" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    assert "the switch waits until qwen3.5:4b (~3.4 GB) is downloaded" in lines[0].getMessage()
    assert ollama.requests == ["/api/tags"]


def test_worker_startup_switches_once_the_models_are_served(tmp_path, ollama, caplog):
    cfg = _config(tmp_path, ollama.host)
    ollama.models = sorted(SERVED)
    with caplog.at_level(logging.INFO, logger="agenthandover_worker"):
        assert worker_main._migrate_vlm_config(cfg) is True
    lines = [r for r in caplog.records if "[vlm]" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "was 0.4.0's default" in lines[0].getMessage()
    assert mp.upgrade_plan(cfg.read_text(encoding="utf-8"))[0] is None

    # Nothing left to switch: Ollama is not asked again.
    ollama.requests.clear()
    assert worker_main._migrate_vlm_config(cfg) is False
    assert ollama.requests == []


def test_worker_startup_with_ollama_down_keeps_a_cloud_configs_models(tmp_path, caplog):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        V040_CLOUD_AFTER_LOCAL.replace(V040_BLOCK, V040_BLOCK + 'ollama_host = "http://127.0.0.1:9"\n'),
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO, logger="agenthandover_worker"):
        assert worker_main._migrate_vlm_config(cfg) is True
    vlm = tomllib.loads(cfg.read_text(encoding="utf-8"))["vlm"]
    assert vlm["mode"] == "local" and "provider" not in vlm
    assert (vlm["annotation_model"], vlm["sop_model"]) == GEMMA_12B
    lines = [r for r in caplog.records if "[vlm]" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    assert "cloud mode was removed" in lines[0].getMessage()

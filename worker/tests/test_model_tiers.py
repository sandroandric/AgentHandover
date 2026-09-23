"""The one RAM tier table (``model_profiles.TIERS``) and its ``tiers.json``.

``resources/tiers.json`` is what the Rust crates compile in and the Swift
tests decode; the installer writes the same bytes to the data directory for
the app.  These pins keep every copy the same object as the Python table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agenthandover_worker import model_profiles as mp

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED = REPO_ROOT / "resources" / "tiers.json"


def test_committed_tiers_json_is_the_python_table():
    assert COMMITTED.read_text(encoding="utf-8") == mp.tiers_json(), (
        "resources/tiers.json is stale: run "
        "`python -m agenthandover_worker.model_profiles write-tiers resources/tiers.json`"
    )
    assert json.loads(COMMITTED.read_text()) == mp.tiers_document()


def test_write_tiers_writes_the_committed_bytes(tmp_path):
    target = tmp_path / "data" / "tiers.json"
    subprocess.run(
        [sys.executable, "-m", "agenthandover_worker.model_profiles",
         "write-tiers", str(target)],
        check=True,
    )
    assert target.read_bytes() == COMMITTED.read_bytes()


def test_vlm_block_command_prints_the_block():
    out = subprocess.run(
        [sys.executable, "-m", "agenthandover_worker.model_profiles",
         "vlm-block", "recommended"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert out == (
        '[vlm]\nmode = "local"\ntier = "recommended"\n'
        'annotation_model = "qwen3.5:4b"\nsop_model = "gemma4:12b-it-qat"\n'
    )
    assert out == mp.vlm_block(mp.get_tier("recommended"))


def test_unknown_tier_is_an_error():
    proc = subprocess.run(
        [sys.executable, "-m", "agenthandover_worker.model_profiles",
         "vlm-block", "huge"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2 and "huge" in proc.stderr


@pytest.mark.parametrize(("ram", "tier"), [
    (0, "standard"), (8, "standard"), (15, "standard"),
    (16, "recommended"), (18, "recommended"), (24, "performance"),
    (36, "performance"), (48, "max_quality"), (128, "max_quality"),
])
def test_recommend_tier_by_ram(ram, tier):
    assert mp.recommend_tier(ram).id == tier


def test_the_measured_split():
    """The annotator is qwen3.5:4b on every tier; Skills and synthesis run on
    gemma4:12b-it-qat at 16/24 GB and gemma4:31b-it-qat at 48 GB."""
    assert {t.annotation_model for t in mp.TIERS} == {"qwen3.5:4b"}
    assert {t.id: t.sop_model for t in mp.TIERS} == {
        "standard": "qwen3.5:4b",
        "recommended": "gemma4:12b-it-qat",
        "performance": "gemma4:12b-it-qat",
        "max_quality": "gemma4:31b-it-qat",
    }
    assert mp.get_tier("max_quality").ab_sop_model == "gemma4:26b-a4b-it-qat"
    assert [t.min_ram_gb for t in mp.TIERS] == sorted(t.min_ram_gb for t in mp.TIERS)
    assert mp.TIERS[0].min_ram_gb == 0


def test_every_tier_model_has_a_registered_profile_and_a_size():
    for tier in mp.TIERS:
        for model in (tier.annotation_model, tier.sop_model, tier.ab_sop_model):
            if model:
                assert model in mp._PROFILES, model
        for model in tier.models:
            assert model in mp.MODEL_SIZES_GB, model
    assert mp.get_tier("standard").models == ["qwen3.5:4b", "nomic-embed-text"]


def test_minimum_ollama_version():
    assert mp.MINIMUM_OLLAMA_VERSION_FOR_GEMMA4 == "0.34.0"
    assert mp.version_at_least("0.34.2", "0.34.0")
    assert mp.version_at_least("0.34.0-rc1", "0.34.0")
    assert not mp.version_at_least("0.30.6", "0.34.0")
    assert not mp.version_at_least("garbage", "0.34.0")


def test_machine_ram_is_whole_gib():
    ram = mp.machine_ram_gb()
    assert isinstance(ram, int) and ram > 0
    assert mp.recommend_tier().id == mp.recommend_tier(ram).id

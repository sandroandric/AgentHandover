"""Removing ``[embedding] image_embeddings`` (WP-E1).

The SigLIP "Visual search" option is gone in 0.6.0.  The worker's startup
migration (``model_profiles.migrate_embedding_file``) removes the key from
config.toml after a timestamped backup, and when it was on, first records
``{"wanted_visual_search": true}`` in ``memory-offer.json`` beside the
worker's status files, where the app's Memory offer reads it.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from agenthandover_worker import main as worker_main
from agenthandover_worker import model_profiles as mp
from agenthandover_worker.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "vlm_block"
NOW = datetime(2026, 10, 12, 9, 30, 0)

# What 0.5's onboarding left after "Pull All Models" with Visual search on:
# the whole [embedding] section rewritten at the end of the file.
ONBOARDED_ON = (
    '[vlm]\nmode = "local"\ntier = "standard"\n'
    'annotation_model = "qwen3.5:2b-q4_K_M"\nsop_model = "qwen3.5:4b"\n'
    '\n[embedding]\nmodel = "nomic-embed-text"\nimage_embeddings = true\n'
)
# A hand-edited section: the key between the user's other keys, with an
# inline comment, and an unrelated section after it.
HAND_EDITED = (
    '[embedding]\nmodel = "nomic-embed-text"\nretention_days = 365\n'
    "image_embeddings = true   # visual search\nmax_rows = 80000\n"
    '\n[privacy]\nimage_embeddings = "not ours"\n'
)


def _offer(state_dir: Path) -> dict | None:
    path = state_dir / mp.MEMORY_OFFER_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --- the text edit ---------------------------------------------------------


def test_v040_onboarding_file_loses_only_the_key_line():
    text = (FIXTURES / "v0.4.0-onboarding.toml").read_text(encoding="utf-8")
    new, wanted = mp.drop_image_embeddings(text)
    assert wanted is False
    assert new == text.replace("image_embeddings = false\n", "")


def test_the_installers_note_goes_with_the_key():
    text = (FIXTURES / "recommended-previous.toml").read_text(encoding="utf-8")
    new, wanted = mp.drop_image_embeddings(text)
    assert wanted is False
    assert new == text.replace(
        "# Image embeddings need the optional embeddings-apple extra.\n"
        "image_embeddings = false\n", "",
    )
    assert new.endswith('[embedding]\nmodel = "nomic-embed-text"\n')


def test_other_keys_and_sections_are_kept_byte_for_byte():
    new, wanted = mp.drop_image_embeddings(HAND_EDITED)
    assert wanted is True
    assert new == HAND_EDITED.replace("image_embeddings = true   # visual search\n", "")
    # A key of the same name in another section is not ours.
    assert 'image_embeddings = "not ours"' in new


@pytest.mark.parametrize(("value", "wanted"), [
    ("true", True), ("false", False), ("1", True), ("0", False),
    ('"true"', True), ('"no"', False), ('"maybe"', False),
])
def test_on_is_what_the_config_loader_read(value, wanted):
    new, got = mp.drop_image_embeddings(f"[embedding]\nimage_embeddings = {value}\n")
    assert (new, got) == ("[embedding]\n", wanted)


@pytest.mark.parametrize("text", [
    "",
    '[embedding]\nmodel = "nomic-embed-text"\n',
    "[vlm]\nimage_embeddings = true\n",
    # Forms a line edit cannot remove safely are left alone.
    "embedding = { image_embeddings = true }\n",
    "embedding.image_embeddings = true\n[embedding.x]\n",
    "[embedding]\nimage_embeddings = true\n[broken\n",
])
def test_nothing_to_remove_or_unsafe_changes_nothing(text):
    assert mp.drop_image_embeddings(text) == (text, None)


def test_a_key_given_twice_by_dotted_form_is_left_alone():
    text = "[embedding]\nmodel = \"m\"\n\n[embedding.extra]\nimage_embeddings = true\n"
    assert mp.drop_image_embeddings(text) == (text, None)


# --- the file migration ----------------------------------------------------


def test_on_is_recorded_then_removed_with_a_backup(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(ONBOARDED_ON, encoding="utf-8")
    cfg.chmod(0o600)
    state = tmp_path / "state"

    changed, wanted, backup = mp.migrate_embedding_file(cfg, state, now=NOW)

    assert (changed, wanted) == (True, True)
    assert backup == tmp_path / "config.toml.20261012-093000.bak"
    assert backup.read_text(encoding="utf-8") == ONBOARDED_ON
    assert cfg.read_text(encoding="utf-8") == ONBOARDED_ON.replace("image_embeddings = true\n", "")
    assert cfg.stat().st_mode & 0o777 == 0o600
    assert _offer(state) == {"wanted_visual_search": True}
    assert not hasattr(load_config(cfg).embedding, "image_embeddings")


def test_off_is_removed_and_records_nothing(tmp_path):
    cfg = tmp_path / "config.toml"
    v040 = (FIXTURES / "v0.4.0-onboarding.toml").read_text(encoding="utf-8")
    cfg.write_text(v040, encoding="utf-8")

    changed, wanted, backup = mp.migrate_embedding_file(cfg, tmp_path, now=NOW)

    assert (changed, wanted) == (True, False)
    assert backup is not None and backup.read_text(encoding="utf-8") == v040
    assert "image_embeddings" not in cfg.read_text(encoding="utf-8")
    assert _offer(tmp_path) is None


def test_a_second_run_changes_nothing(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(ONBOARDED_ON, encoding="utf-8")
    mp.migrate_embedding_file(cfg, tmp_path, now=NOW)
    after = cfg.read_text(encoding="utf-8")

    assert mp.migrate_embedding_file(cfg, tmp_path, now=NOW) == (False, False, None)
    assert cfg.read_text(encoding="utf-8") == after
    assert len(list(tmp_path.glob("config.toml.*.bak"))) == 1
    assert _offer(tmp_path) == {"wanted_visual_search": True}


def test_the_offer_files_other_keys_are_kept(tmp_path):
    (tmp_path / mp.MEMORY_OFFER_FILE).write_text('{"offered": false}', encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(ONBOARDED_ON, encoding="utf-8")
    mp.migrate_embedding_file(cfg, tmp_path, now=NOW)
    assert _offer(tmp_path) == {"offered": False, "wanted_visual_search": True}


def test_an_unreadable_offer_file_is_replaced(tmp_path):
    (tmp_path / mp.MEMORY_OFFER_FILE).write_text("[1, 2", encoding="utf-8")
    cfg = tmp_path / "config.toml"
    cfg.write_text(ONBOARDED_ON, encoding="utf-8")
    mp.migrate_embedding_file(cfg, tmp_path, now=NOW)
    assert _offer(tmp_path) == {"wanted_visual_search": True}


def test_a_symlinked_config_is_written_through(tmp_path):
    real = tmp_path / "dotfiles" / "config.toml"
    real.parent.mkdir()
    real.write_text(ONBOARDED_ON, encoding="utf-8")
    link = tmp_path / "config.toml"
    link.symlink_to(real)
    changed, _, backup = mp.migrate_embedding_file(link, tmp_path, now=NOW)
    assert changed and link.is_symlink()
    assert backup.parent == real.parent
    assert "image_embeddings" not in real.read_text(encoding="utf-8")


# --- worker startup and the CLI ---------------------------------------------


def test_worker_startup_migrates_with_one_info_line(tmp_path, caplog):
    cfg = tmp_path / "config.toml"
    cfg.write_text(ONBOARDED_ON, encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="agenthandover_worker"):
        assert worker_main._migrate_embedding_config(cfg, tmp_path) is True
    lines = [r for r in caplog.records if "image_embeddings" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "(it was on;" in lines[0].getMessage()
    assert _offer(tmp_path) == {"wanted_visual_search": True}

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="agenthandover_worker"):
        assert worker_main._migrate_embedding_config(cfg, tmp_path) is False
    assert [r for r in caplog.records if "image_embeddings" in r.getMessage()] == []


def test_worker_startup_records_in_the_status_dir(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(ONBOARDED_ON, encoding="utf-8")
    status = tmp_path / "status"
    monkeypatch.setattr(worker_main, "_status_dir", lambda: status)
    assert worker_main._migrate_embedding_config(cfg) is True
    assert _offer(status) == {"wanted_visual_search": True}


def test_worker_startup_without_a_config_does_nothing(tmp_path):
    assert worker_main._migrate_embedding_config(tmp_path / "config.toml", tmp_path) is False
    assert list(tmp_path.iterdir()) == []


def test_worker_startup_warns_and_continues_on_a_bad_file(tmp_path, caplog):
    cfg = tmp_path / "config.toml"
    cfg.write_bytes(b"[embedding]\nimage_embeddings = true\n\xff\xfe\n")
    with caplog.at_level(logging.WARNING, logger="agenthandover_worker"):
        assert worker_main._migrate_embedding_config(cfg, tmp_path) is False
    assert any(
        r.levelno == logging.WARNING and "Could not migrate [embedding]" in r.getMessage()
        for r in caplog.records
    )


def test_cli_migrate_embedding(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(ONBOARDED_ON, encoding="utf-8")
    run = [sys.executable, "-m", "agenthandover_worker.model_profiles",
           "migrate-embedding", str(cfg), str(tmp_path)]
    first = subprocess.run(run, capture_output=True, text=True, check=True)
    assert f"Removed [embedding] image_embeddings from {cfg.resolve()} (it was on)" in first.stdout
    assert _offer(tmp_path) == {"wanted_visual_search": True}
    second = subprocess.run(run, capture_output=True, text=True, check=True)
    assert second.stdout == ""
    missing = subprocess.run(
        [*run[:4], str(tmp_path / "none.toml"), str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert "no [embedding] keys to migrate" in missing.stdout

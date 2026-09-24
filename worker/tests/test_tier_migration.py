"""Tier migration: a config still on a tier's earlier default models
(``model_profiles.PREVIOUS_TIER_DEFAULTS``) moves to the tier's current
ones at worker startup, once Ollama serves them (the installer's
``migrate-vlm`` asks no Ollama and keeps them); a block the user
customised is never touched.

The shared fixtures in ``tests/fixtures/vlm_block`` pin the edit to the
one ``agenthandover setup`` (``tiers::apply_vlm_block``) and onboarding
(``ModelTiers.applyVLMBlock``) make.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from agenthandover_worker import main as worker_main
from agenthandover_worker import model_profiles as mp

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "vlm_block"

OLD_RECOMMENDED = FIXTURES.joinpath("recommended-previous.toml").read_text(encoding="utf-8")
MIGRATED_RECOMMENDED = FIXTURES.joinpath("recommended-migrated.toml").read_text(encoding="utf-8")
OLD_STANDARD = FIXTURES.joinpath("standard-previous.toml").read_text(encoding="utf-8")
MIGRATED_STANDARD = FIXTURES.joinpath("standard-migrated.toml").read_text(encoding="utf-8")

# What Ollama's /api/tags lists on a Mac with every tier's models pulled.
SERVED = frozenset(m for t in mp.TIERS for m in t.models)


def _unchanged(text: str) -> str:
    new, changed, reason = mp.migrate_vlm_block(text, SERVED)
    assert not changed, reason
    assert new == text
    return reason


# --- the history ----------------------------------------------------------


def test_history_names_known_tiers_and_never_the_current_pair():
    for tier_id, pairs in mp.PREVIOUS_TIER_DEFAULTS.items():
        tier = mp.get_tier(tier_id)
        assert tier is not None, tier_id
        assert pairs and len(set(pairs)) == len(pairs)
        assert (tier.annotation_model, tier.sop_model) not in pairs


def test_history_is_the_wp_f6_and_wp_f8_changes():
    """The defaults that changed since WP-C6: the 16 GB tier's Skills
    model, gemma4:12b-it-qat until dfaa35d (WP-F6), and the 8 GB tier's
    annotator, qwen3.5:4b until WP-F8."""
    assert mp.PREVIOUS_TIER_DEFAULTS == {
        "standard": (("qwen3.5:4b", "qwen3.5:4b"),),
        "recommended": (("qwen3.5:4b", "gemma4:12b-it-qat"),),
    }


def test_tiers_json_carries_the_history():
    doc = mp.tiers_document()
    by_id = {t["id"]: t["previous_defaults"] for t in doc["tiers"]}
    assert by_id == {
        "standard": [{"annotation_model": "qwen3.5:4b", "sop_model": "qwen3.5:4b"}],
        "recommended": [{"annotation_model": "qwen3.5:4b", "sop_model": "gemma4:12b-it-qat"}],
        "performance": [],
        "max_quality": [],
    }


# --- apply_vlm_block: the third writer, pinned to the shared fixture ------


def test_apply_matches_the_shared_fixture():
    tier = mp.get_tier("performance")
    source = FIXTURES.joinpath("input.toml").read_text(encoding="utf-8")
    expected = FIXTURES.joinpath("performance.toml").read_text(encoding="utf-8")
    once = mp.apply_vlm_block(source, tier)
    assert once == expected
    assert mp.apply_vlm_block(once, tier) == once


def test_apply_appends_to_a_config_without_vlm():
    tier = mp.get_tier("standard")
    assert mp.apply_vlm_block("[embedding]\nimage_embeddings = false", tier) == (
        "[embedding]\nimage_embeddings = false\n\n" + mp.vlm_block(tier)
    )
    assert mp.apply_vlm_block("", tier) == "\n" + mp.vlm_block(tier)


# --- migrate_vlm_block -----------------------------------------------------


def test_old_recommended_block_moves_to_qwen_byte_for_byte():
    new, changed, reason = mp.migrate_vlm_block(OLD_RECOMMENDED, SERVED)
    assert changed
    assert new == MIGRATED_RECOMMENDED
    # Only the sop_model line differs; every other byte is kept.
    old_lines = OLD_RECOMMENDED.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    assert len(old_lines) == len(new_lines)
    diff = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
    assert diff == [('sop_model = "gemma4:12b-it-qat"\n', 'sop_model = "qwen3.5:4b"\n')]
    assert reason.startswith('tier "recommended": sop_model gemma4:12b-it-qat -> qwen3.5:4b')
    # What `agenthandover setup --tier recommended` writes.
    assert new == mp.apply_vlm_block(OLD_RECOMMENDED, mp.get_tier("recommended"))


def test_old_standard_block_moves_to_the_2b_annotator_byte_for_byte():
    """WP-F8: an 8 GB config the installer or onboarding wrote with
    qwen3.5:4b for both roles now annotates with qwen3.5:2b-q4_K_M; only
    the annotation_model line changes."""
    new, changed, reason = mp.migrate_vlm_block(OLD_STANDARD, SERVED)
    assert changed
    assert new == MIGRATED_STANDARD
    diff = [
        (a, b) for a, b in zip(OLD_STANDARD.splitlines(True), new.splitlines(True)) if a != b
    ]
    assert diff == [('annotation_model = "qwen3.5:4b"\n', 'annotation_model = "qwen3.5:2b-q4_K_M"\n')]
    assert reason == (
        'tier "standard": annotation_model qwen3.5:4b -> qwen3.5:2b-q4_K_M '
        "(qwen3.5:4b + qwen3.5:4b was the tier's earlier default; "
        "it now runs qwen3.5:2b-q4_K_M + qwen3.5:4b)"
    )
    assert new == mp.apply_vlm_block(OLD_STANDARD, mp.get_tier("standard"))
    # Idempotent.
    assert _unchanged(new) == 'already the current models of tier "standard"'


@pytest.mark.parametrize(("label", "text", "reason"), [
    ("customised sop_model",
     OLD_STANDARD.replace('sop_model = "qwen3.5:4b"', 'sop_model = "gemma4:12b-it-qat"'),
     'qwen3.5:4b + gemma4:12b-it-qat is not an earlier default of tier "standard" (set by hand)'),
    ("customised annotation_model",
     OLD_STANDARD.replace('annotation_model = "qwen3.5:4b"', 'annotation_model = "qwen3.5:2b"'),
     'qwen3.5:2b + qwen3.5:4b is not an earlier default of tier "standard" (set by hand)'),
    ("the old standard pair on the recommended tier",
     OLD_STANDARD.replace('tier = "standard"', 'tier = "recommended"'),
     'already the current models of tier "recommended"'),
    ("annotation_model left to the tier",
     OLD_STANDARD.replace('annotation_model = "qwen3.5:4b"\n', ""),
     "[vlm] does not pin both models (the tier supplies the rest)"),
])
def test_a_customised_standard_block_is_left_alone(label, text, reason):
    assert _unchanged(text) == reason, label


def test_migration_is_idempotent():
    once, changed, _ = mp.migrate_vlm_block(OLD_RECOMMENDED, SERVED)
    assert changed
    reason = _unchanged(once)
    assert reason == 'already the current models of tier "recommended"'


def test_the_current_default_is_a_no_op():
    for tier in mp.TIERS:
        text = "[export]\nadapter = \"all\"\n\n" + mp.vlm_block(tier) + "ollama_host = \"x\"\n"
        assert _unchanged(text) == f'already the current models of tier "{tier.id}"'


def test_extra_vlm_keys_and_sections_survive():
    text = (
        "# my config\n"
        "[vlm]\n"
        'ollama_host = "http://127.0.0.1:11500"\n'
        'mode = "local"\n'
        'tier = "recommended"\n'
        'annotation_model = "qwen3.5:4b"\n'
        'sop_model = "gemma4:12b-it-qat"\n'
        "sop_ab = false\n"
        "\n"
        "[privacy]\n"
        "read_app_state = true\n"
    )
    new, changed, _ = mp.migrate_vlm_block(text, SERVED)
    assert changed
    # The managed keys go right after the header, as every writer puts
    # them; the hand-set keys and the other section are kept verbatim.
    assert new == (
        "# my config\n"
        "[vlm]\n"
        'mode = "local"\n'
        'tier = "recommended"\n'
        'annotation_model = "qwen3.5:4b"\n'
        'sop_model = "qwen3.5:4b"\n'
        'ollama_host = "http://127.0.0.1:11500"\n'
        "sop_ab = false\n"
        "\n"
        "[privacy]\n"
        "read_app_state = true\n"
    )


def test_block_without_mode_is_local_and_migrates():
    text = OLD_RECOMMENDED.replace('mode = "local"\n', "")
    new, changed, _ = mp.migrate_vlm_block(text, SERVED)
    assert changed
    # The rewritten block states the mode, as every writer does.
    assert new == MIGRATED_RECOMMENDED


def test_crlf_and_no_trailing_newline_are_kept():
    text = '[vlm]\r\nmode = "local"\r\ntier = "recommended"\r\nannotation_model = "qwen3.5:4b"\r\nsop_model = "gemma4:12b-it-qat"\r\nkeep = 1'
    new, changed, _ = mp.migrate_vlm_block(text, SERVED)
    assert changed
    assert new.endswith("keep = 1")
    assert "sop_model" in new and "gemma4" not in new


@pytest.mark.parametrize(("label", "text", "reason"), [
    ("customised sop_model",
     OLD_RECOMMENDED.replace('sop_model = "gemma4:12b-it-qat"', 'sop_model = "gemma4:31b-it-qat"'),
     'qwen3.5:4b + gemma4:31b-it-qat is not an earlier default of tier "recommended" (set by hand)'),
    ("customised annotation_model",
     OLD_RECOMMENDED.replace('annotation_model = "qwen3.5:4b"', 'annotation_model = "qwen3.5:2b"'),
     'qwen3.5:2b + gemma4:12b-it-qat is not an earlier default of tier "recommended" (set by hand)'),
    ("unknown tier",
     OLD_RECOMMENDED.replace('tier = "recommended"', 'tier = "huge"'),
     'unknown tier "huge"'),
    ("the old pair under another tier",
     OLD_RECOMMENDED.replace('tier = "recommended"', 'tier = "performance"'),
     'already the current models of tier "performance"'),
    ("no tier key",
     OLD_RECOMMENDED.replace('tier = "recommended"\n', ""),
     "[vlm] names no tier and qwen3.5:4b + gemma4:12b-it-qat is not a default of 0.4.0 "
     "(set by hand)"),
    ("sop_model left to the tier",
     OLD_RECOMMENDED.replace('sop_model = "gemma4:12b-it-qat"\n', ""),
     "[vlm] does not pin both models (the tier supplies the rest)"),
    ("comment on a managed line",
     OLD_RECOMMENDED.replace('sop_model = "gemma4:12b-it-qat"', 'sop_model = "gemma4:12b-it-qat"  # keep gemma'),
     '[vlm] sop_model was edited by hand (sop_model = "gemma4:12b-it-qat"  # keep gemma)'),
    ("single-quoted value",
     OLD_RECOMMENDED.replace('sop_model = "gemma4:12b-it-qat"', "sop_model = 'gemma4:12b-it-qat'"),
     "[vlm] sop_model was edited by hand (sop_model = 'gemma4:12b-it-qat')"),
    ("invalid TOML elsewhere",
     OLD_RECOMMENDED + "[broken\n",
     None),
    ("no [vlm] section",
     '[export]\nadapter = "all"\n',
     "no [vlm] section"),
    ("empty config", "", "no [vlm] section"),
])
def test_left_alone(label, text, reason):
    got = _unchanged(text)
    if reason is None:
        assert got.startswith("config.toml is not valid TOML"), got
    else:
        assert got == reason, label


def test_a_duplicate_managed_key_is_left_alone():
    text = OLD_RECOMMENDED.replace(
        'sop_model = "gemma4:12b-it-qat"\n',
        'sop_model = "gemma4:12b-it-qat"\nsop_model = "gemma4:12b-it-qat"\n',
    )
    assert _unchanged(text) == "[vlm] sets sop_model more than once"


def test_managed_keys_in_a_later_section_are_not_vlms():
    # sop_model under [other] is not [vlm]'s: [vlm] pins no sop_model.
    text = (
        '[vlm]\nmode = "local"\ntier = "recommended"\nannotation_model = "qwen3.5:4b"\n'
        '\n[other]\nsop_model = "gemma4:12b-it-qat"\n'
    )
    assert _unchanged(text) == "[vlm] does not pin both models (the tier supplies the rest)"


# --- migrate_vlm_file ------------------------------------------------------

NOW = datetime(2026, 9, 23, 20, 15, 0)


def test_file_is_backed_up_then_rewritten(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(OLD_RECOMMENDED, encoding="utf-8")
    cfg.chmod(0o600)
    changed, reason, backup = mp.migrate_vlm_file(cfg, now=NOW, served=SERVED)
    assert changed and "gemma4:12b-it-qat -> qwen3.5:4b" in reason
    assert backup == tmp_path / "config.toml.20260923-201500.bak"
    assert backup.read_text(encoding="utf-8") == OLD_RECOMMENDED
    assert cfg.read_text(encoding="utf-8") == MIGRATED_RECOMMENDED
    assert oct(cfg.stat().st_mode & 0o777) == oct(0o600)
    assert oct(backup.stat().st_mode & 0o777) == oct(0o600)
    # No temporary file left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "config.toml", "config.toml.20260923-201500.bak",
    ]


def test_no_backup_and_no_write_when_unchanged(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(MIGRATED_RECOMMENDED, encoding="utf-8")
    before = cfg.stat()
    changed, reason, backup = mp.migrate_vlm_file(cfg, now=NOW, served=SERVED)
    assert (changed, backup) == (False, None)
    assert reason == 'already the current models of tier "recommended"'
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]
    after = cfg.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_second_run_is_a_no_op_and_writes_no_second_backup(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(OLD_RECOMMENDED, encoding="utf-8")
    assert mp.migrate_vlm_file(cfg, now=NOW, served=SERVED)[0]
    assert mp.migrate_vlm_file(cfg, now=NOW, served=SERVED)[0] is False
    assert len(list(tmp_path.glob("config.toml.*.bak"))) == 1


def test_backup_name_does_not_overwrite_an_earlier_one(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(OLD_RECOMMENDED, encoding="utf-8")
    earlier = tmp_path / "config.toml.20260923-201500.bak"
    earlier.write_text("earlier backup\n", encoding="utf-8")
    _, _, backup = mp.migrate_vlm_file(cfg, now=NOW, served=SERVED)
    assert backup == tmp_path / "config.toml.20260923-201500-2.bak"
    assert earlier.read_text(encoding="utf-8") == "earlier backup\n"


def test_a_symlinked_config_is_written_through(tmp_path):
    real = tmp_path / "dotfiles" / "config.toml"
    real.parent.mkdir()
    real.write_text(OLD_RECOMMENDED, encoding="utf-8")
    link = tmp_path / "config.toml"
    link.symlink_to(real)
    changed, _, backup = mp.migrate_vlm_file(link, now=NOW, served=SERVED)
    assert changed
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == MIGRATED_RECOMMENDED
    assert backup.parent == real.parent


# --- the installer's subcommand -------------------------------------------


def _migrate_vlm(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agenthandover_worker.model_profiles", "migrate-vlm", str(path)],
        capture_output=True, text=True,
    )


def test_migrate_vlm_command_keeps_the_models_and_says_why(tmp_path):
    """The installer asks no Ollama, so it cannot know the new models are
    pulled: it keeps the old ones and says what the switch waits for."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(OLD_RECOMMENDED, encoding="utf-8")
    proc = _migrate_vlm(cfg)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == (
        f"Kept [vlm] in {cfg} as it is: tier \"recommended\": "
        "sop_model gemma4:12b-it-qat -> qwen3.5:4b (qwen3.5:4b + gemma4:12b-it-qat was "
        "the tier's earlier default; it now runs qwen3.5:4b + qwen3.5:4b); the switch "
        "waits until Ollama serves qwen3.5:4b (the app offers the download; or: "
        "agenthandover setup --tier recommended)\n"
    )
    assert cfg.read_text(encoding="utf-8") == OLD_RECOMMENDED
    assert list(tmp_path.glob("config.toml.*.bak")) == []

    cfg.write_text(MIGRATED_RECOMMENDED, encoding="utf-8")
    again = _migrate_vlm(cfg)
    assert again.returncode == 0
    assert again.stdout == (
        f"Kept [vlm] in {cfg} as it is: already the current models of tier \"recommended\"\n"
    )
    assert list(tmp_path.glob("config.toml.*.bak")) == []


def test_migrate_vlm_command_without_a_config(tmp_path):
    proc = _migrate_vlm(tmp_path / "config.toml")
    assert proc.returncode == 0
    assert proc.stdout == f"No {tmp_path / 'config.toml'}; no [vlm] models to migrate\n"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_migrate_vlm_command_fails_cleanly_on_an_unreadable_config(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(OLD_RECOMMENDED, encoding="utf-8")
    cfg.chmod(0)
    try:
        proc = _migrate_vlm(cfg)
    finally:
        cfg.chmod(0o600)
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr.startswith(f"could not migrate the [vlm] models in {cfg}: ")
    assert cfg.read_text(encoding="utf-8") == OLD_RECOMMENDED


# --- worker startup --------------------------------------------------------


def test_worker_startup_migrates_with_one_info_line(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(worker_main, "served_models", lambda text: sorted(SERVED))
    cfg = tmp_path / "config.toml"
    cfg.write_text(OLD_RECOMMENDED, encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="agenthandover_worker"):
        assert worker_main._migrate_vlm_config(cfg) is True
    lines = [r for r in caplog.records if "[vlm]" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "sop_model gemma4:12b-it-qat -> qwen3.5:4b" in lines[0].getMessage()
    assert cfg.read_text(encoding="utf-8") == MIGRATED_RECOMMENDED

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="agenthandover_worker"):
        assert worker_main._migrate_vlm_config(cfg) is False
    assert [r for r in caplog.records if "[vlm]" in r.getMessage()] == []


def test_worker_startup_without_a_config_does_nothing(tmp_path):
    assert worker_main._migrate_vlm_config(tmp_path / "config.toml") is False
    assert list(tmp_path.iterdir()) == []


def test_worker_startup_warns_and_continues_on_a_bad_file(tmp_path, caplog):
    cfg = tmp_path / "config.toml"
    cfg.write_bytes(b"[vlm]\nmode = \"local\"\n\xff\xfe\n")
    with caplog.at_level(logging.WARNING, logger="agenthandover_worker"):
        assert worker_main._migrate_vlm_config(cfg) is False
    assert any(
        r.levelno == logging.WARNING and "Could not check the [vlm] models" in r.getMessage()
        for r in caplog.records
    )
    assert cfg.read_bytes() == b"[vlm]\nmode = \"local\"\n\xff\xfe\n"

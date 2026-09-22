"""Tests for the VLM setup CLI module."""

from __future__ import annotations

import sys

from agenthandover_worker.setup_vlm import (
    check_vlm_available,
    detect_platform,
    install_vlm,
    main,
    prompt_install,
)


class TestDetectPlatform:
    def test_returns_valid_string(self) -> None:
        result = detect_platform()
        assert result in ("apple_silicon", "macos_intel", "linux", "other")


class TestCheckVLMAvailable:
    def test_returns_dict_with_expected_keys(self) -> None:
        result = check_vlm_available()
        assert isinstance(result, dict)
        assert set(result) == {"ollama", "openai_compat"}
        assert isinstance(result["ollama"], bool)
        assert isinstance(result["openai_compat"], bool)


class TestPromptInstall:
    def test_non_interactive_declines(self, monkeypatch) -> None:
        """Non-interactive mode (not a TTY) should skip install."""
        devnull = open("/dev/null")
        try:
            monkeypatch.setattr("sys.stdin", devnull)
            result = prompt_install("apple_silicon", {"ollama": False, "openai_compat": False})
            assert result is False
        finally:
            devnull.close()

    def test_already_reachable_skips(self, capsys) -> None:
        """If Ollama is already reachable, skip the install prompt."""
        result = prompt_install(
            "apple_silicon", {"ollama": True, "openai_compat": False}
        )
        assert result is False
        captured = capsys.readouterr()
        assert "already reachable" in captured.out


class TestInstallVLM:
    def test_dry_run_builds_correct_command(self, capsys) -> None:
        cmd = install_vlm("vlm-ollama", dry_run=True)
        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "pip"
        assert cmd[3] == "install"
        assert "agenthandover-worker[vlm-ollama]" in cmd[4]
        captured = capsys.readouterr()
        assert "Would run" in captured.out


class TestMain:
    def test_check_flag(self, capsys) -> None:
        """--check should print VLM status and exit."""
        try:
            main(["--check"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "Platform:" in captured.out
        assert "ollama available:" in captured.out
        assert "openai-compat available:" in captured.out
        assert "VLM ready:" in captured.out

    def test_dry_run_flag(self, capsys) -> None:
        """--dry-run should show command without executing."""
        main(["--dry-run"])
        captured = capsys.readouterr()
        assert "Would run" in captured.out
        assert "agenthandover-worker[vlm-ollama]" in captured.out

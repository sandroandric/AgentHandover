"""Tests for the VLM setup CLI module."""

from __future__ import annotations

import sys

from oc_apprentice_worker.setup_vlm import (
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
        assert "mlx_vlm" in result
        assert "llama_cpp" in result
        assert isinstance(result["mlx_vlm"], bool)
        assert isinstance(result["llama_cpp"], bool)


class TestPromptInstall:
    def test_non_interactive_declines(self, monkeypatch) -> None:
        """Non-interactive mode (not a TTY) should skip install."""
        devnull = open("/dev/null")
        try:
            monkeypatch.setattr("sys.stdin", devnull)
            result = prompt_install("apple_silicon", {"mlx_vlm": False, "llama_cpp": False})
            assert result is False
        finally:
            devnull.close()

    def test_already_installed_skips(self, capsys) -> None:
        """If the recommended backend is already installed, skip prompt."""
        result = prompt_install(
            "apple_silicon", {"mlx_vlm": True, "llama_cpp": False}
        )
        assert result is False
        captured = capsys.readouterr()
        assert "already installed" in captured.out

    def test_non_apple_already_installed_skips(self, capsys) -> None:
        result = prompt_install(
            "linux", {"mlx_vlm": False, "llama_cpp": True}
        )
        assert result is False
        captured = capsys.readouterr()
        assert "already installed" in captured.out


class TestInstallVLM:
    def test_dry_run_builds_correct_command(self, capsys) -> None:
        cmd = install_vlm("vlm-apple", dry_run=True)
        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "pip"
        assert cmd[3] == "install"
        assert "oc-apprentice-worker[vlm-apple]" in cmd[4]
        captured = capsys.readouterr()
        assert "Would run" in captured.out

    def test_cpu_extras_dry_run(self, capsys) -> None:
        cmd = install_vlm("vlm-cpu", dry_run=True)
        assert "oc-apprentice-worker[vlm-cpu]" in cmd[4]


class TestMain:
    def test_check_flag(self, capsys) -> None:
        """--check should print VLM status and exit."""
        try:
            main(["--check"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "Platform:" in captured.out
        assert "VLM ready:" in captured.out

    def test_dry_run_flag(self, capsys) -> None:
        """--dry-run should show command without executing."""
        main(["--dry-run"])
        captured = capsys.readouterr()
        assert "Would run" in captured.out

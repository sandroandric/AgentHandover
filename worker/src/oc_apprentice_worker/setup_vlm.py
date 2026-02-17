"""Interactive VLM setup CLI for OpenMimic.

Detects the platform, checks VLM availability, and offers to install
the recommended VLM backend. Designed to be run as:
    oc-setup-vlm
or:
    python -m oc_apprentice_worker.setup_vlm
"""

from __future__ import annotations

import platform
import subprocess
import sys


def detect_platform() -> str:
    """Detect the current platform for VLM backend selection.

    Returns one of: 'apple_silicon', 'macos_intel', 'linux', 'other'.
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        if machine == "arm64":
            return "apple_silicon"
        return "macos_intel"
    if system == "Linux":
        return "linux"
    return "other"


def check_vlm_available() -> dict[str, bool]:
    """Check which VLM backends are importable.

    Returns a dict with keys for each backend, each True/False.
    """
    result: dict[str, bool] = {
        "mlx_vlm": False,
        "llama_cpp": False,
        "ollama": False,
        "openai_compat": False,
    }

    try:
        import mlx_vlm  # noqa: F401
        result["mlx_vlm"] = True
    except ImportError:
        pass

    try:
        import llama_cpp  # noqa: F401
        result["llama_cpp"] = True
    except ImportError:
        pass

    try:
        import ollama
        ollama.Client().list()
        result["ollama"] = True
    except Exception:
        pass

    try:
        import openai  # noqa: F401
        import os
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENMIMIC_API_KEY"):
            result["openai_compat"] = True
    except ImportError:
        pass

    return result


def _recommend_extras(plat: str) -> str:
    """Return the recommended pip extras group for the given platform."""
    if plat == "apple_silicon":
        return "vlm-apple"
    return "vlm-cpu"


def prompt_install(plat: str, available: dict[str, bool]) -> bool:
    """Prompt the user to install VLM dependencies.

    Returns True if the user confirmed installation, False otherwise.
    In non-interactive mode (not a TTY), returns False.
    """
    extras = _recommend_extras(plat)

    # Check if already available (before TTY check so callers always get this info)
    if plat == "apple_silicon" and available["mlx_vlm"]:
        print("mlx-vlm is already installed. VLM is ready.")
        return False
    if plat != "apple_silicon" and available["llama_cpp"]:
        print("llama-cpp-python is already installed. VLM is ready.")
        return False

    if not sys.stdin.isatty():
        print("Non-interactive mode detected — skipping VLM install prompt.")
        return False

    print()
    print("=" * 60)
    print("  OpenMimic VLM Setup")
    print("=" * 60)
    print()
    print(f"  Platform detected: {plat}")
    print(f"  Recommended extras: {extras}")
    print()
    print("  VLM (Vision Language Model) enables better observation")
    print("  of native apps by analyzing screenshots with AI.")
    print("  It is RECOMMENDED for the best experience.")
    print()
    print("  Additional backends: pip install oc-apprentice-worker[vlm-ollama]")
    print("                       pip install oc-apprentice-worker[vlm-openai]")
    print()

    try:
        answer = input("  Install VLM backend? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    return answer in ("", "y", "yes")


def install_vlm(extras: str, *, dry_run: bool = False) -> list[str]:
    """Build and optionally run the pip install command.

    Returns the command as a list of strings.
    If dry_run is True, prints the command but does not execute it.
    """
    cmd = [sys.executable, "-m", "pip", "install", f"oc-apprentice-worker[{extras}]"]

    if dry_run:
        print(f"  Would run: {' '.join(cmd)}")
        return cmd

    print(f"  Running: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n  Installation failed (exit code {result.returncode}).")
        print("  You can try manually: pip install " + f"oc-apprentice-worker[{extras}]")
    else:
        print("\n  VLM backend installed successfully.")

    return cmd


def main(argv: list[str] | None = None) -> None:
    """Entry point for oc-setup-vlm."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="oc-setup-vlm",
        description="Set up VLM backend for OpenMimic",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without running pip",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check VLM availability and exit",
    )
    args = parser.parse_args(argv)

    plat = detect_platform()
    available = check_vlm_available()

    if args.check:
        print(f"Platform: {plat}")
        print(f"mlx-vlm available: {available['mlx_vlm']}")
        print(f"llama-cpp-python available: {available['llama_cpp']}")
        print(f"ollama available: {available['ollama']}")
        print(f"openai-compat available: {available['openai_compat']}")
        any_available = any(available.values())
        print(f"VLM ready: {any_available}")
        sys.exit(0 if any_available else 1)

    extras = _recommend_extras(plat)

    if args.dry_run:
        install_vlm(extras, dry_run=True)
        return

    if prompt_install(plat, available):
        install_vlm(extras)

        # Verify installation
        print("\n  Verifying installation...")
        new_available = check_vlm_available()
        any_ready = any(new_available.values())
        if any_ready:
            print("  VLM is now available.")
        else:
            print("  VLM not detected after install. You may need to restart your shell.")


if __name__ == "__main__":
    main()

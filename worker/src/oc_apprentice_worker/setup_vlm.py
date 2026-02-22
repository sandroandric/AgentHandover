"""Interactive VLM setup CLI for OpenMimic.

Detects the platform, checks VLM availability, and offers to install
the recommended VLM backend. Supports local and remote (cloud API)
setups. Designed to be run as:
    oc-setup-vlm
or:
    python -m oc_apprentice_worker.setup_vlm
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


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
        # Mark openai_compat available only when BOTH an API key is set
        # AND the base URL points to a local server (deny_network_egress).
        has_key = bool(
            os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENMIMIC_API_KEY")
        )
        base_url = os.environ.get("OPENMIMIC_VLM_BASE_URL", "")
        _local_prefixes = (
            "http://localhost", "http://127.0.0.1",
            "https://localhost", "https://127.0.0.1",
            "http://[::1]",
        )
        is_local = base_url and any(base_url.startswith(p) for p in _local_prefixes)
        if has_key and is_local:
            result["openai_compat"] = True
    except ImportError:
        pass

    return result


# Remote cloud provider metadata
_REMOTE_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "key_prefix": "sk-",
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-20250514",
        "key_prefix": "sk-ant-",
    },
    "google": {
        "name": "Google (Gemini)",
        "env_var": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash",
        "key_prefix": "AI",
    },
}


def _config_path() -> Path:
    """Return the OS-appropriate config.toml path."""
    if platform.system() == "Darwin":
        return (
            Path.home() / "Library" / "Application Support"
            / "oc-apprentice" / "config.toml"
        )
    return Path.home() / ".config" / "oc-apprentice" / "config.toml"


def validate_api_key(provider: str, api_key: str) -> bool:
    """Lightweight validation that the API key format looks correct.

    Does NOT make an actual API call — just checks prefix and length.
    """
    if not api_key or len(api_key) < 10:
        return False
    meta = _REMOTE_PROVIDERS.get(provider)
    if meta and meta["key_prefix"]:
        return api_key.startswith(meta["key_prefix"])
    return True


def prompt_remote_setup() -> dict | None:
    """Interactive remote VLM setup: provider picker, privacy consent, key input.

    Returns a dict with keys {provider, model, api_key_env, env_var_value}
    or None if the user cancels.
    """
    if not sys.stdin.isatty():
        print("Non-interactive mode — skipping remote setup.")
        return None

    print()
    print("=" * 60)
    print("  OpenMimic Remote VLM Setup")
    print("=" * 60)
    print()
    print("  ⚠️  PRIVACY WARNING")
    print("  Remote VLM mode sends screenshots of your desktop to a")
    print("  cloud API for analysis. Only enable this if you understand")
    print("  and accept this privacy trade-off.")
    print()

    try:
        consent = input("  Do you consent to sending screenshots to a cloud API? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if consent not in ("y", "yes"):
        print("  Remote VLM setup cancelled.")
        return None

    # Provider selection
    print()
    print("  Available providers:")
    providers = list(_REMOTE_PROVIDERS.keys())
    for i, key in enumerate(providers, 1):
        meta = _REMOTE_PROVIDERS[key]
        print(f"    {i}. {meta['name']} (default model: {meta['default_model']})")

    print()
    try:
        choice = input(f"  Select provider [1-{len(providers)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(providers)):
            raise ValueError
    except (ValueError, IndexError):
        print("  Invalid selection.")
        return None

    provider = providers[idx]
    meta = _REMOTE_PROVIDERS[provider]

    # Check if env var already set
    env_var = meta["env_var"]
    existing_key = os.environ.get(env_var, "")

    if existing_key:
        print(f"\n  ✓ {env_var} already set in environment.")
        use_existing = input("  Use existing key? [Y/n] ").strip().lower()
        if use_existing in ("", "y", "yes"):
            return {
                "provider": provider,
                "model": meta["default_model"],
                "api_key_env": env_var,
                "env_var_value": existing_key,
            }

    # Prompt for API key
    print(f"\n  Enter your {meta['name']} API key:")
    print(f"  (expected prefix: {meta['key_prefix']}...)")
    try:
        api_key = input("  API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not validate_api_key(provider, api_key):
        print("  ⚠️  Key format looks incorrect, but proceeding anyway.")

    print(f"\n  To persist the key, add to your shell profile:")
    print(f"    export {env_var}=\"{api_key[:8]}...\"")

    return {
        "provider": provider,
        "model": meta["default_model"],
        "api_key_env": env_var,
        "env_var_value": api_key,
    }


def _write_remote_config(result: dict) -> None:
    """Update config.toml [vlm] section with remote mode settings.

    Reads the existing TOML, updates the [vlm] fields, and writes back.
    If the file doesn't exist, creates it with just the [vlm] section.
    """
    import tomllib

    config_file = _config_path()
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing config (preserve other sections)
    existing_lines: list[str] = []
    if config_file.is_file():
        existing_lines = config_file.read_text(encoding="utf-8").splitlines(keepends=True)

    # Strategy: find [vlm] section and insert/update mode/provider/model/api_key_env
    # Simple approach: rewrite the file with updated [vlm] fields
    cfg: dict = {}
    if config_file.is_file():
        with open(config_file, "rb") as f:
            cfg = tomllib.load(f)

    vlm = cfg.get("vlm", {})
    vlm["mode"] = "remote"
    vlm["provider"] = result["provider"]
    vlm["model"] = result["model"]
    vlm["api_key_env"] = result["api_key_env"]
    cfg["vlm"] = vlm

    # Write back preserving readability
    # We'll rebuild TOML manually to keep comments from example
    # For simplicity, just write the updated fields
    _write_toml_config(config_file, cfg)

    print(f"\n  ✓ Config written to {config_file}")


def _write_toml_config(path: Path, cfg: dict) -> None:
    """Write a dict as TOML to *path*.

    Uses a simple serializer (no third-party TOML writer needed).
    """
    lines: list[str] = []
    # Write top-level simple keys first (none expected, but just in case)
    for key, val in cfg.items():
        if not isinstance(val, dict):
            lines.append(f"{key} = {_toml_value(val)}")

    # Write sections
    for section, values in cfg.items():
        if isinstance(values, dict):
            lines.append(f"\n[{section}]")
            for key, val in values.items():
                lines.append(f"{key} = {_toml_value(val)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _toml_value(val: object) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return str(val)
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, list):
        items = ", ".join(_toml_value(v) for v in val)
        return f"[{items}]"
    return f'"{val}"'


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
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Set up a remote cloud VLM provider (OpenAI, Anthropic, Google)",
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

    # Remote setup mode
    if args.remote:
        result = prompt_remote_setup()
        if result is not None:
            _write_remote_config(result)
            print("\n  Remote VLM setup complete!")
            print("  Restart the worker to activate: openmimic restart worker")
        return

    extras = _recommend_extras(plat)

    if args.dry_run:
        install_vlm(extras, dry_run=True)
        return

    # Offer remote option before local install
    if not any(available.values()) and sys.stdin.isatty():
        print()
        print("  No local VLM backend detected.")
        print("  You can either install a local backend or use a cloud API.")
        print()
        try:
            use_remote = input("  Set up a cloud VLM provider instead? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            use_remote = "n"

        if use_remote in ("y", "yes"):
            result = prompt_remote_setup()
            if result is not None:
                _write_remote_config(result)
                print("\n  Remote VLM setup complete!")
                print("  Restart the worker to activate: openmimic restart worker")
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

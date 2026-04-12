#!/usr/bin/env bash
#
# install-native-host.sh — Install the Chrome Native Messaging host manifest
# for the AgentHandover daemon (com.agenthandover.host).
#
# Usage:
#   ./scripts/install-native-host.sh [--extension-id <ID>] [--daemon-path <PATH>]
#
# Options:
#   --extension-id <ID>    Chrome extension ID (default: uses wildcard for dev)
#   --daemon-path  <PATH>  Absolute path to agenthandover-daemon binary
#                          (default: searches cargo target/release, then target/debug)

set -euo pipefail

HOST_NAME="com.agenthandover.host"
MANIFEST_FILE="${HOST_NAME}.json"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

EXTENSION_ID=""
DAEMON_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extension-id)
            EXTENSION_ID="$2"
            shift 2
            ;;
        --daemon-path)
            DAEMON_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--extension-id <ID>] [--daemon-path <PATH>]"
            echo ""
            echo "Install the Chrome Native Messaging host manifest for"
            echo "the AgentHandover daemon."
            echo ""
            echo "Options:"
            echo "  --extension-id <ID>    Chrome extension ID"
            echo "  --daemon-path  <PATH>  Absolute path to the daemon binary"
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Detect OS and set manifest directory
# ---------------------------------------------------------------------------

detect_manifest_dir() {
    local os
    os="$(uname -s)"

    case "$os" in
        Darwin)
            echo "${HOME}/Library/Application Support/Google/Chrome/NativeMessagingHosts"
            ;;
        Linux)
            echo "${HOME}/.config/google-chrome/NativeMessagingHosts"
            ;;
        *)
            echo "Error: unsupported operating system: ${os}" >&2
            echo "Only macOS (Darwin) and Linux are supported." >&2
            exit 1
            ;;
    esac
}

MANIFEST_DIR="$(detect_manifest_dir)"

# ---------------------------------------------------------------------------
# Locate daemon binary
# ---------------------------------------------------------------------------

if [[ -z "$DAEMON_PATH" ]]; then
    # Try to find the binary in the project's cargo target directories.
    # Priority: universal binary (just build-all) > release > debug.
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

    if [[ -x "${PROJECT_ROOT}/target/universal-release/agenthandover-daemon" ]]; then
        DAEMON_PATH="${PROJECT_ROOT}/target/universal-release/agenthandover-daemon"
    elif [[ -x "${PROJECT_ROOT}/target/release/agenthandover-daemon" ]]; then
        DAEMON_PATH="${PROJECT_ROOT}/target/release/agenthandover-daemon"
    elif [[ -x "${PROJECT_ROOT}/target/debug/agenthandover-daemon" ]]; then
        DAEMON_PATH="${PROJECT_ROOT}/target/debug/agenthandover-daemon"
    else
        echo "Error: could not find agenthandover-daemon binary." >&2
        echo "Build first with: just build-all (or cargo build --release)" >&2
        echo "Or specify with: $0 --daemon-path /path/to/agenthandover-daemon" >&2
        exit 1
    fi
fi

# Resolve to absolute path
DAEMON_PATH="$(cd "$(dirname "$DAEMON_PATH")" && pwd)/$(basename "$DAEMON_PATH")"

if [[ ! -x "$DAEMON_PATH" ]]; then
    echo "Error: daemon binary not found or not executable: ${DAEMON_PATH}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build allowed_origins
# ---------------------------------------------------------------------------

if [[ -z "$EXTENSION_ID" ]]; then
    # Use the stable extension ID derived from the RSA key in manifest.json.
    # Override via --extension-id if you build/sign the extension yourself.
    EXTENSION_ID="jpemkdcihaijkolbkankcldmiimmmnfo"
    echo "Using default extension ID: ${EXTENSION_ID}"
fi
ALLOWED_ORIGINS="[\"chrome-extension://${EXTENSION_ID}/\"]"

# ---------------------------------------------------------------------------
# Generate and install the manifest
# ---------------------------------------------------------------------------

mkdir -p "$MANIFEST_DIR"

MANIFEST_PATH="${MANIFEST_DIR}/${MANIFEST_FILE}"

cat > "$MANIFEST_PATH" <<MANIFEST_EOF
{
  "name": "${HOST_NAME}",
  "description": "AgentHandover Observer Bridge",
  "path": "${DAEMON_PATH}",
  "type": "stdio",
  "allowed_origins": ${ALLOWED_ORIGINS}
}
MANIFEST_EOF

echo "Native messaging host manifest installed successfully."
echo ""
echo "  Host name:    ${HOST_NAME}"
echo "  Manifest:     ${MANIFEST_PATH}"
echo "  Daemon path:  ${DAEMON_PATH}"
echo "  Extension ID: ${EXTENSION_ID}"
echo ""
echo "To verify, check: cat \"${MANIFEST_PATH}\""

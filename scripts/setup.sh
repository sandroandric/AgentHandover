#!/usr/bin/env bash
# Unified setup script for AgentHandover.
# Installs native messaging host, then pulls the local models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stable extension ID from the RSA key in extension/manifest.json.
# Override with: AGENTHANDOVER_EXTENSION_ID=<id> ./scripts/setup.sh
EXTENSION_ID="${AGENTHANDOVER_EXTENSION_ID:-jpemkdcihaijkolbkankcldmiimmmnfo}"

echo "=== AgentHandover Setup ==="
echo

# Step 1: Install Chrome native messaging host
if [ -f "$SCRIPT_DIR/install-native-host.sh" ]; then
    echo "Step 1/2: Installing Chrome native messaging host..."
    bash "$SCRIPT_DIR/install-native-host.sh" --extension-id "$EXTENSION_ID"
    echo
else
    echo "Step 1/2: install-native-host.sh not found — skipping."
    echo
fi

# Step 2: the local models (Ollama) of this Mac's RAM tier
echo "Step 2/2: Local models..."
if command -v agenthandover >/dev/null 2>&1; then
    agenthandover setup --vlm "$@"
else
    echo "  agenthandover CLI not found; after installing it run: agenthandover setup --vlm"
fi

echo
echo "=== Setup complete ==="

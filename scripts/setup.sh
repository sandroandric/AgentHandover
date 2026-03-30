#!/usr/bin/env bash
# Unified setup script for AgentHandover.
# Installs native messaging host, then offers VLM setup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stable extension ID from the RSA key in extension/manifest.json.
# Override with: AGENTHANDOVER_EXTENSION_ID=<id> ./scripts/setup.sh
EXTENSION_ID="${AGENTHANDOVER_EXTENSION_ID:-knldjmfmopnpolahpmmgbagdohdnhkik}"

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

# Step 2: VLM setup (use worker venv if available)
echo "Step 2/2: VLM setup..."
WORKER_VENV="$SCRIPT_DIR/../worker/.venv/bin/python"
if [ -x "$WORKER_VENV" ]; then
    "$WORKER_VENV" -m agenthandover_worker.setup_vlm "$@"
else
    python3 -m agenthandover_worker.setup_vlm "$@"
fi

echo
echo "=== Setup complete ==="

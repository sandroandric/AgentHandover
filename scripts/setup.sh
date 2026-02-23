#!/usr/bin/env bash
# Unified setup script for OpenMimic.
# Installs native messaging host, then offers VLM setup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stable extension ID from the RSA key in extension/manifest.json.
# Override with: OPENMIMIC_EXTENSION_ID=<id> ./scripts/setup.sh
EXTENSION_ID="${OPENMIMIC_EXTENSION_ID:-knldjmfmopnpolahpmmgbagdohdnhkik}"

echo "=== OpenMimic Setup ==="
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

# Step 2: VLM setup
echo "Step 2/2: VLM setup..."
python -m oc_apprentice_worker.setup_vlm "$@"

echo
echo "=== Setup complete ==="

#!/usr/bin/env bash
set -euo pipefail

echo "=== AgentHandover Uninstaller ==="
echo ""

# Stop services
echo "Stopping services..."
launchctl unload ~/Library/LaunchAgents/com.agenthandover.daemon.plist 2>/dev/null || true
launchctl unload ~/Library/LaunchAgents/com.agenthandover.worker.plist 2>/dev/null || true

# Remove launchd plists
echo "Removing launchd plists..."
rm -f ~/Library/LaunchAgents/com.agenthandover.daemon.plist
rm -f ~/Library/LaunchAgents/com.agenthandover.worker.plist

# Remove binaries
echo "Removing binaries..."
sudo rm -f /usr/local/bin/agenthandover
# Note: agenthandover-daemon lives inside /Applications/AgentHandover.app
# (removed below with the app bundle)

# Remove native messaging host
echo "Removing native messaging host..."
rm -f ~/Library/Application\ Support/Google/Chrome/NativeMessagingHosts/com.agenthandover.host.json

# Remove lib directory (venv, extension, etc.)
echo "Removing library files..."
sudo rm -rf /usr/local/lib/agenthandover

# Remove app bundle
echo "Removing app..."
rm -rf /Applications/AgentHandover.app

# Remove PID and status files
echo "Removing runtime files..."
rm -f ~/Library/Application\ Support/agenthandover/daemon.pid
rm -f ~/Library/Application\ Support/agenthandover/worker.pid
rm -f ~/Library/Application\ Support/agenthandover/daemon-status.json
rm -f ~/Library/Application\ Support/agenthandover/worker-status.json

echo ""
echo "Uninstall complete."
echo ""
echo "User data preserved at: ~/Library/Application Support/agenthandover/"
echo "  (database, config, logs)"
echo ""
echo "To also remove user data, run:"
echo "  rm -rf ~/Library/Application\\ Support/agenthandover/"

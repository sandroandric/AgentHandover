import SwiftUI

struct MenuBarView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var delegate: AppDelegate
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header with status
            headerSection

            Divider()

            // Stats
            statsSection

            Divider()

            // Service controls
            controlsSection

            Divider()

            // Quick actions
            actionsSection

            Divider()

            // Quit
            Button("Quit OpenMimic") {
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q")
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .frame(width: 300)
        .onChange(of: delegate.pendingOnboarding) { pending in
            if pending {
                delegate.pendingOnboarding = false
                openWindow(id: "onboarding")
            }
        }
        .onAppear {
            // Also check on first appear (when user clicks menu bar icon)
            if !hasCompletedOnboarding && delegate.pendingOnboarding {
                delegate.pendingOnboarding = false
                openWindow(id: "onboarding")
            }
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(appState.health.color)
                .frame(width: 10, height: 10)

            VStack(alignment: .leading, spacing: 2) {
                Text("OpenMimic")
                    .font(.headline)
                Text(appState.health.label)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text("v\(appState.daemonVersion)")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            StatRow(label: "Events Today", value: "\(appState.eventsToday)")
            StatRow(label: "SOPs Generated", value: "\(appState.sopsGenerated)")

            // VLM queue status (only shown when VLM is available)
            if appState.workerStatus?.vlm_available == true {
                let pending = appState.vlmQueuePending
                if pending > 0 {
                    StatRow(
                        label: "VLM Queue",
                        value: appState.vlmBacklogged
                            ? "\(pending) pending ⚠️"
                            : "\(pending) pending"
                    )
                }
            }

            HStack(spacing: 12) {
                ServicePill(
                    name: "Daemon",
                    running: appState.daemonRunning
                )
                ServicePill(
                    name: "Worker",
                    running: appState.workerRunning
                )
                ServicePill(
                    name: "Extension",
                    running: appState.extensionConnected
                )
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var controlsSection: some View {
        VStack(spacing: 4) {
            if appState.daemonRunning || appState.workerRunning {
                Button(action: {
                    appState.userStopped = true
                    ServiceController.stopAll()
                }) {
                    Label("Stop All Services", systemImage: "stop.circle")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            } else {
                Button(action: {
                    appState.userStopped = false
                    ServiceController.startAll()
                }) {
                    Label("Start All Services", systemImage: "play.circle")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            }

            Button(action: {
                ServiceController.restartAll()
            }) {
                Label("Restart Services", systemImage: "arrow.clockwise.circle")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
        }
        .padding(.vertical, 4)
    }

    private var actionsSection: some View {
        VStack(spacing: 4) {
            // Open config
            Button(action: openConfig) {
                Label("Edit Configuration", systemImage: "gearshape")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)

            // Open SOPs directory
            Button(action: openSOPsDir) {
                Label("Open SOPs Folder", systemImage: "folder")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)

            // View logs
            Button(action: openLogs) {
                Label("View Logs", systemImage: "doc.text")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)

            // Setup / Permissions / Extension CTA
            if !hasCompletedOnboarding {
                Button(action: {
                    openWindow(id: "onboarding")
                }) {
                    Label("Complete Setup", systemImage: "sparkles")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundColor(.blue)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            } else if !appState.accessibilityGranted || !appState.screenRecordingGranted {
                Button(action: {
                    openWindow(id: "onboarding")
                }) {
                    Label("Fix Permissions", systemImage: "exclamationmark.shield")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundColor(.orange)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            } else if !appState.extensionConnected {
                Button(action: {
                    openWindow(id: "onboarding")
                }) {
                    Label("Connect Extension", systemImage: "puzzlepiece.extension")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundColor(.orange)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: - Actions

    private func openConfig() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configPath = home
            .appendingPathComponent("Library/Application Support/oc-apprentice/config.toml")
        NSWorkspace.shared.open(configPath)
    }

    private func openSOPsDir() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let sopsPath = home
            .appendingPathComponent(".openclaw/workspace/memory/apprentice/sops")
        if FileManager.default.fileExists(atPath: sopsPath.path) {
            NSWorkspace.shared.open(sopsPath)
        } else {
            // Fall back to the apprentice data dir
            let dataDir = home
                .appendingPathComponent("Library/Application Support/oc-apprentice")
            NSWorkspace.shared.open(dataDir)
        }
    }

    private func openLogs() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let logsPath = home
            .appendingPathComponent("Library/Application Support/oc-apprentice/logs")
        NSWorkspace.shared.open(logsPath)
    }
}

// MARK: - Subviews

struct StatRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }
}

struct ServicePill: View {
    let name: String
    let running: Bool

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(running ? Color.green : Color.red)
                .frame(width: 6, height: 6)
            Text(name)
                .font(.caption2)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(Color.secondary.opacity(0.1))
        )
    }
}
